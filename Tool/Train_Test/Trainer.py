"""
Trainer.py -- Single-GPU training with memory optimizations.

Key features:
  - BF16 AMP with GradScaler fallback for FP16
  - Gradient NaN/Inf detection and skip
  - Checkpoint save/resume
  - Multi-dataset training with per-type ratio sampling
  - RAM optimization: uint8 images in DataLoader, float conversion on GPU
"""

import os
import gc
import tqdm
import torch
import numpy as np
import torch.nn as nn
import random
import time
import hashlib
import warnings
from collections import defaultdict

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from Tool.Datasets.VODataest import PoseDataset, VODataset
from Tool.Datasets.transformation import motion_ses2pose_quats
from Tool.Evaluator.tartanair_evaluator import TartanAirEvaluator
from Tool.Utils.load_save import get_checkpoint_state, load_checkpoint, save_checkpoint
from Tool.Utils.utils import my_worker_init_fn, apply_dataset_ratio
from Tool.Utils.optimizer import build_optimizer
from Tool.Utils.scheduler import build_lr_scheduler
from Tool.Datasets.multi_dataset import (
    build_multi_train_dataset,
    build_test_transform,
    get_intrinsic_for,
)


def _build_multi_val_dataset(paths_dict, target_size=(480, 640), logger=None):
    """Build a concatenated multi-dataset validation set."""
    test_imgs = paths_dict['test_img']
    test_flows = paths_dict['test_flow']
    test_poses = paths_dict['test_pose']
    test_types = paths_dict['test_types']

    groups = defaultdict(lambda: ([], [], []))
    for i, t in enumerate(test_types):
        groups[t][0].append(test_imgs[i])
        groups[t][1].append(test_flows[i] if i < len(test_flows) else None)
        groups[t][2].append(test_poses[i])

    sub_datasets = []
    for ds_type, (imgs, flows, poses) in groups.items():
        transform = build_test_transform(ds_type, target_size=target_size)
        try:
            vo = VODataset(imgs, flows, None, poses, transform=transform, keep_uint8=True)
        except TypeError:
            vo = VODataset(imgs, flows, None, poses, transform=transform)
        vo.dataset_type = ds_type
        sub_datasets.append(vo)
        if logger is not None:
            fx, fy, cx, cy = get_intrinsic_for(ds_type)
            logger.info(
                f"[MultiDataset] val / {ds_type}: {len(vo)} samples  "
                f"(fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f})")

    if not sub_datasets:
        raise RuntimeError("No validation sub-datasets built. Check test_datasets in YAML.")

    concat = ConcatDataset(sub_datasets)
    if logger is not None:
        logger.info(f"[MultiDataset] val total: {len(concat)} samples "
                    f"across {len(sub_datasets)} sub-datasets")
    return concat


class Trainer(object):
    def __init__(self, cfg, model, logger, loss, paths_dict, writer, model_name='MVOFormer',
                 tester=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = build_optimizer(cfg['optimizer'], model)
        self.lr_scheduler, self.warmup_lr_scheduler = build_lr_scheduler(
            cfg['lr_scheduler'], self.optimizer, last_epoch=-1)

        self.logger = logger
        self.writer = writer

        self.epoch = 0
        self.best_result = 100
        self.best_epoch = 0
        self.v_loss_final = 1
        self.v_loss_pose = 1
        self.v_loss_un = 1

        self.device = next(self.model.parameters()).device
        self.pose_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['trainer']['save_path'], model_name)
        self.tester = tester

        self.batch_size = cfg['dataset']['batch_size']
        self.num_workers = cfg['trainer']['num_workers']
        self.scale = cfg['trainer']['scale']
        self.paths_dict = paths_dict
        self.step_train_datasets = cfg['trainer']['step']
        self.grad_clip = float(cfg.get('trainer', {}).get('grad_clip', 1.0))
        self.logger.info(f"grad_clip max_norm = {self.grad_clip}")

        # AMP mixed precision setup
        amp_dtype_str = cfg.get('trainer', {}).get('amp_dtype', 'bf16').lower()

        if amp_dtype_str == 'bf16' and torch.cuda.is_bf16_supported():
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            self.amp_dtype_str = 'bf16'
            self.scaler = None
        elif amp_dtype_str == 'fp16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
            self.amp_dtype_str = 'fp16'
            self.scaler = GradScaler('cuda')
        else:
            if amp_dtype_str == 'bf16' and not torch.cuda.is_bf16_supported():
                warnings.warn("GPU does not support BF16, falling back to FP16 + GradScaler.")
                self.amp_enabled = True
                self.amp_dtype = torch.float16
                self.amp_dtype_str = 'fp16 (fallback)'
                self.scaler = GradScaler('cuda')
            else:
                self.amp_enabled = False
                self.amp_dtype = torch.float32
                self.amp_dtype_str = 'fp32 (disabled)'
                self.scaler = None

        self.logger.info(f"AMP: {self.amp_dtype_str}  "
                         f"GradScaler: {'enabled' if self.scaler else 'disabled'}")

        # Load pretrain / resume checkpoint
        if cfg.get('trainer', {}).get('pretrain_model', None):
            assert os.path.exists(cfg['trainer']['pretrain_model'])
            load_checkpoint(model=self.model, optimizer=None,
                            filename=cfg['trainer']['pretrain_model'],
                            map_location=self.device, logger=self.logger)

        if cfg.get('trainer', {}).get('resume_model', None):
            resume_model_path = cfg['trainer']['resume_model']
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch, self.pose_losses_test = load_checkpoint(
                model=self.model, optimizer=self.optimizer,
                filename=resume_model_path, map_location=self.device,
                logger=self.logger, scaler=self.scaler)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info(f"Resumed checkpoint. Best Result: {self.best_result}, "
                             f"Best Epoch: {self.best_epoch}")

        # Multi-dataset DataLoader construction
        target_size = tuple(cfg.get('dataset', {}).get('target_size', (480, 640)))
        max_scale = cfg.get('dataset', {}).get('max_scale', 2.5)

        # Training set
        train_ratio = cfg['dataset'].get('train_ratio', 1.0)
        paths_for_train = paths_dict
        if train_ratio < 1.0:
            paths_for_train = self._apply_ratio_per_type(
                paths_dict, ratio=train_ratio, seed=cfg.get('random_seed', 666))
            self.logger.info(
                f"[Dataset] train_ratio={train_ratio:.2f}  "
                f"samples: {len(paths_for_train['train_img'])} "
                f"/ {len(paths_dict['train_img'])}")

        aug_cfg = cfg.get('dataset', {}).get('augmentation', None)
        self.trainDataset = build_multi_train_dataset(
            paths_for_train, step=self.step_train_datasets,
            target_size=target_size, max_scale=max_scale,
            aug_cfg=aug_cfg, logger=self.logger)

        # DataLoader params from config
        prefetch_factor = int(self.cfg.get('trainer', {}).get('prefetch_factor', 1))
        pin_memory_train = bool(self.cfg.get('trainer', {}).get('pin_memory_train', False))

        self.train_loader = DataLoader(
            self.trainDataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=prefetch_factor if self.num_workers > 0 else None,
            worker_init_fn=my_worker_init_fn, drop_last=True,
            persistent_workers=False, pin_memory=pin_memory_train)
        self.logger.info(
            f"[train_loader] batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}, prefetch_factor={prefetch_factor}, "
            f"pin_memory={pin_memory_train}")

        # Validation set
        self.valid_Dataset = _build_multi_val_dataset(
            paths_dict, target_size=target_size, logger=self.logger)
        val_num_workers = int(self.cfg.get('trainer', {}).get('val_num_workers', 2))
        val_num_workers = min(val_num_workers, self.num_workers)
        val_batch_size = int(self.cfg.get('trainer', {}).get('val_batch_size', self.batch_size))

        self.val_num_workers = val_num_workers
        self.val_batch_size = val_batch_size

        self.valid_dataloader = DataLoader(
            self.valid_Dataset, batch_size=val_batch_size, shuffle=False,
            num_workers=val_num_workers,
            prefetch_factor=1 if val_num_workers > 0 else None,
            drop_last=False, persistent_workers=False, pin_memory=False)
        self.logger.info(
            f"[val_loader] batch_size={val_batch_size}, "
            f"num_workers={val_num_workers}, pin_memory=False")

    @staticmethod
    def _apply_ratio_per_type(paths_dict, ratio, seed=666):
        """Per-type stratified sampling with deterministic hash-based seed."""
        train_types = paths_dict['train_types']
        idx_by_type = defaultdict(list)
        for i, t in enumerate(train_types):
            idx_by_type[t].append(i)

        keep_idx = []
        for type_idx, t in enumerate(sorted(idx_by_type.keys())):
            idxs = idx_by_type[t]
            stable_hash = int(hashlib.md5(str(t).encode('utf-8')).hexdigest()[:8], 16)
            rng_local = random.Random(seed + type_idx * 10_000 + stable_hash % 10_000)
            idxs_shuf = idxs.copy()
            rng_local.shuffle(idxs_shuf)
            n_keep = max(1, int(round(len(idxs_shuf) * ratio)))
            keep_idx.extend(idxs_shuf[:n_keep])
        keep_idx.sort()

        new_paths = dict(paths_dict)
        for k in ['train_img', 'train_flow', 'train_mask', 'train_pose', 'train_types']:
            src = paths_dict.get(k, [])
            new_paths[k] = [src[i] for i in keep_idx if i < len(src)] if src else []
        return new_paths

    def _to_device(self, data):
        """Move data to GPU and convert uint8 images to float32.

        Dataset returns uint8 img1/img2 to save RAM. Conversion to float32
        happens on GPU, avoiding overhead in CPU workers.
        """
        out = {}
        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                out[k] = v
                continue
            v = v.to(self.device, non_blocking=True)
            if (k in ('img1', 'img2')) and v.dtype == torch.uint8:
                v = v.float()
            out[k] = v
        return out

    # ==================================================================
    # Main training loop
    # ==================================================================
    def train(self):
        start_epoch = self.epoch
        self.cfg_trian = self.cfg['trainer']

        progress_bar = tqdm.tqdm(
            range(start_epoch, self.cfg_trian['max_epoch']),
            dynamic_ncols=True, leave=True,
            desc=f'epoch {start_epoch + 1}/{self.cfg_trian["max_epoch"]}')

        best_result = self.best_result
        best_epoch = self.best_epoch

        for epoch in range(start_epoch, self.cfg_trian['max_epoch']):
            progress_bar.set_description(
                f'epoch {epoch + 1}/{self.cfg_trian["max_epoch"]}')

            np.random.seed(np.random.get_state()[1][0] + epoch)

            pose_losses_dict_log_epoch, pose_losses_dict_log = self.train_one_epoch(epoch)
            self.epoch += 1

            _warmup_epochs = int(self.cfg['lr_scheduler'].get('warmup_epochs', 3))
            if self.warmup_lr_scheduler is not None and epoch < _warmup_epochs:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if (self.epoch % self.cfg_trian['save_frequency']) == 0:
                self.val_loss()

                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg_trian['save_all']:
                    ckpt_name = os.path.join(self.output_dir,
                                             'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')

                scaler_state = self.scaler.state_dict() if self.scaler is not None else None

                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch,
                                         best_result, best_epoch,
                                         scaler_state=scaler_state),
                    ckpt_name)

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    self.tester.paths_dict = self.paths_dict
                    if self.v_loss_final < best_result:
                        best_result = self.v_loss_final
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(
                                self.model, self.optimizer, self.epoch,
                                best_result, best_epoch, float(self.v_loss_final),
                                scaler_state=scaler_state),
                            ckpt_name)
                    self.logger.info(
                        "Best Result For val_loss:{}, epoch:{}".format(
                            best_result, best_epoch))

            if self.writer is not None:
                self.writer.add_scalars('Loss/Epoch', {
                    'loss_ep': pose_losses_dict_log_epoch.get("loss_final", 0.0),
                    'vloss_ep': self.v_loss_final,
                }, epoch)

            progress_bar.update()

            os.makedirs(self.output_dir, exist_ok=True)
            scaler_state = self.scaler.state_dict() if self.scaler is not None else None
            save_checkpoint(
                get_checkpoint_state(self.model, self.optimizer, self.epoch,
                                     best_result, best_epoch, scaler_state=scaler_state),
                os.path.join(self.output_dir, 'checkpoint_final'))
            self.logger.info(f"[Epoch {self.epoch}] checkpoint_final saved.")

            log_dict = {
                "v_loss_final": self.v_loss_final,
                "v_loss_pose": self.v_loss_pose,
                "v_loss_un": self.v_loss_un,
                "t_loss_final": pose_losses_dict_log_epoch.get("loss_final", 0.0),
                "t_loss_pose": pose_losses_dict_log_epoch.get("loss_pose", 0.0),
                "t_loss_un": pose_losses_dict_log_epoch.get("loss_final_uncertainty", 0.0),
            }
            for key, val in pose_losses_dict_log_epoch.items():
                log_dict[key] = round(val, 5)
            log_items = list(log_dict.items())
            log_msg_epoch = ", ".join([f"{k}={v}" for k, v in log_items])
            print('One Epoch Total Loss: ' + log_msg_epoch)

        if self.writer is not None:
            self.writer.add_scalar('Best Result/Minimum val_loss', best_result, best_epoch)
            self.writer.close()
        self.logger.info(f"Best Result (Minimum val_loss):{best_result}, epoch:{best_epoch}")

    # ==================================================================
    # Single epoch training
    # ==================================================================
    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()

        epoch_print = '\n' + "-" * 15 + "  " + "Epoch: " + str(epoch + 1) + "  " + "-" * 15
        print(epoch_print)

        progress_bar = tqdm.tqdm(
            total=len(self.train_loader), leave=False, desc='iters',
            dynamic_ncols=True, position=0, mininterval=0.1, smoothing=0.1,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]')

        pose_losses_dict_log_epoch = {"loss_pose": 0.0}
        log_msg = ""
        pose_losses_dict_log = {}
        nan_skip_count = 0

        for batch_idx, data in enumerate(self.train_loader):
            data = self._to_device(data)
            img1 = data['img1']
            img2 = data['img2']
            flow = data['flow']
            pose = data['pose']
            intrinsic = data['intrinsic']
            inputs = [img1, img2, flow, intrinsic, pose]

            with autocast('cuda', dtype=self.amp_dtype, enabled=self.amp_enabled):
                outputs = self.model(inputs, rnn_time=False)
                losses_dict = self.pose_loss(outputs, pose, is_aux=True)
                weight_dict = self.pose_loss.weight_dict
                pose_losses_dict_weighted = [
                    losses_dict[k] * weight_dict[k]
                    for k in losses_dict.keys() if k in weight_dict]
                pose_losses = sum(pose_losses_dict_weighted)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(pose_losses).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                pose_losses.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip)
            total_norm_val = float(total_norm)
            grad_is_bad = (
                (total_norm_val != total_norm_val)
                or (total_norm_val == float('inf'))
                or (total_norm_val == float('-inf')))

            if grad_is_bad:
                nan_skip_count += 1
                try:
                    loss_val = pose_losses.item()
                except Exception:
                    loss_val = float('nan')
                self.logger.warning(
                    f"[Epoch {epoch+1} | Batch {batch_idx}] "
                    f"NaN/Inf gradient, skipping optimizer.step(). "
                    f"loss={loss_val:.6f}  grad_norm={total_norm_val}  "
                    f"cumulative skips this epoch: {nan_skip_count}")
                if self.writer is not None:
                    iterate_num = len(self.train_loader) * epoch + batch_idx
                    self.writer.add_scalar('Debug/NaN_skip', 1, iterate_num)
                progress_bar.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    self.scaler.update()
                continue

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            pose_losses_dict_log = {}
            pose_losses_log = 0
            for k in losses_dict.keys():
                if k in weight_dict:
                    pose_losses_dict_log[k] = (losses_dict[k] * weight_dict[k]).item()
                    if k not in pose_losses_dict_log_epoch:
                        pose_losses_dict_log_epoch[k] = 0.0
                    pose_losses_dict_log_epoch[k] += \
                        pose_losses_dict_log[k] / len(self.train_loader)

            loss_pose_total = 0
            loss_final_uncertainty_total = 0
            for k, v in pose_losses_dict_log.items():
                if 'rot' in k or 'trans' in k:
                    loss_pose_total += v
                if 'uncertainty' in k:
                    loss_final_uncertainty_total += v

            pose_losses_dict_log["loss_pose"] = loss_pose_total
            pose_losses_dict_log["loss_final_uncertainty"] = loss_final_uncertainty_total
            pose_losses_dict_log["loss_final"] = loss_pose_total + loss_final_uncertainty_total

            for key in ["loss_pose", "loss_final_uncertainty", "loss_final"]:
                if key not in pose_losses_dict_log_epoch:
                    pose_losses_dict_log_epoch[key] = 0.0
                pose_losses_dict_log_epoch[key] += \
                    pose_losses_dict_log[key] / len(self.train_loader)

            if batch_idx % 10 == 0:
                scaler_info = (f"  scale:{self.scaler.get_scale():.0f}"
                               if self.scaler is not None else "")
                log_dict = {"loss_final": pose_losses_dict_log["loss_final"]}
                for key, val in pose_losses_dict_log.items():
                    log_dict[key] = round(val, 5)

                if self.cfg['model']['aux_loss']:
                    loss_pos = []
                    loss_un = []
                    log_pose_list = ""
                    for i in range(int((len(pose_losses_dict_log) - 6) / 3)):
                        k_trans = 'loss_trans_' + str(i)
                        k_rot = 'loss_rot_' + str(i)
                        k_uncertainty = 'loss_uncertainty_' + str(i)
                        loss_pos.append(round(
                            pose_losses_dict_log[k_trans] + pose_losses_dict_log[k_rot], 5))
                        loss_un.append(round(pose_losses_dict_log[k_uncertainty], 5))
                        log_pose_list += (
                            f"loss_pose_{i}:{loss_pos[i]}  "
                            f"loss_uncertainty_{i}:{loss_un[i]} ")
                    log_msg = (
                        f"loss_final:{round(pose_losses_dict_log['loss_final'], 5)}, "
                        f"loss_pose:{round(pose_losses_dict_log['loss_pose'], 5)}, "
                        f"loss_un:{round(pose_losses_dict_log['loss_final_uncertainty'], 5)}, "
                        + log_pose_list + scaler_info)
                else:
                    log_items = list(log_dict.items())
                    log_msg = ", ".join([f"{k}:{v}" for k, v in log_items]) + scaler_info

            progress_bar.set_postfix_str(log_msg)

            if self.writer is not None:
                iterate_num = len(self.train_loader) * epoch + batch_idx
                self.writer.add_scalars('Loss/Total', {
                    'Loss_final': float(pose_losses_dict_log["loss_final"]),
                    'Loss_uncertainty': float(pose_losses_dict_log['loss_final_uncertainty']),
                    'Loss_pose': float(pose_losses_dict_log['loss_pose']),
                }, iterate_num)
                self.writer.add_scalars('Loss/Pose', {
                    'Loss_pose': float(pose_losses_dict_log["loss_pose"]),
                    'Loss_trans': float(pose_losses_dict_log['loss_trans']),
                    'Loss_rot': float(pose_losses_dict_log['loss_rot']),
                }, iterate_num)
                if self.scaler is not None:
                    self.writer.add_scalar(
                        'AMP/GradScaler_scale', self.scaler.get_scale(), iterate_num)

            progress_bar.update()
            progress_bar.refresh()

        if nan_skip_count > 0:
            self.logger.warning(
                f"[Epoch {epoch+1}] Total {nan_skip_count} NaN/Inf batches skipped, "
                f"({nan_skip_count / len(self.train_loader) * 100:.2f}%).")

        progress_bar.close()
        return pose_losses_dict_log_epoch, pose_losses_dict_log

    # ==================================================================
    def val_loss(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.eval()

        n_batches = len(self.valid_dataloader)
        if n_batches == 0:
            self.logger.warning("[val_loss] valid_dataloader is empty, skip.")
            self.v_loss_final = 0.0
            self.v_loss_pose = 0.0
            self.v_loss_un = 0.0
            self.model.train()
            return

        progress_bar = tqdm.tqdm(
            total=n_batches, leave=True, desc='Evaluation Progress',
            dynamic_ncols=True, position=0, mininterval=0.5, smoothing=0.1)

        sum_final = 0.0
        sum_pose = 0.0
        sum_un = 0.0
        n_seen = 0
        empty_cache_every = 50

        try:
            for batch_idx, data in enumerate(self.valid_dataloader):
                data = self._to_device(data)
                img1 = data['img1']
                img2 = data['img2']
                flow = data['flow']
                pose = data['pose']
                intrinsic = data['intrinsic']
                inputs = [img1, img2, flow, intrinsic, pose]

                with torch.no_grad():
                    with autocast('cuda', dtype=self.amp_dtype, enabled=self.amp_enabled):
                        outputs = self.model(inputs, rnn_time=False)
                    losses_dict = self.pose_loss(outputs, pose, is_aux=False)

                weight_dict = self.pose_loss.weight_dict

                batch_final = 0.0
                batch_pose = 0.0
                batch_un = 0.0
                for k, v in losses_dict.items():
                    if k not in weight_dict:
                        continue
                    w_loss = float((v * weight_dict[k]).detach().item())
                    batch_final += w_loss
                    if 'rot' in k or 'trans' in k:
                        batch_pose += w_loss
                    if 'uncertainty' in k:
                        batch_un += w_loss

                batch_size_actual = img1.size(0)
                sum_final += batch_final * batch_size_actual
                sum_pose += batch_pose * batch_size_actual
                sum_un += batch_un * batch_size_actual
                n_seen += batch_size_actual

                del img1, img2, flow, pose, intrinsic, inputs, data
                del outputs, losses_dict

                if (batch_idx + 1) % empty_cache_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                    avg_final = sum_final / max(n_seen, 1)
                    avg_pose = sum_pose / max(n_seen, 1)
                    avg_un = sum_un / max(n_seen, 1)
                    progress_bar.set_postfix_str(
                        f"final:{avg_final:.4f}, pose:{avg_pose:.4f}, un:{avg_un:.4f}")
                progress_bar.update()

        finally:
            progress_bar.close()
            self.model.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if n_seen == 0:
            self.v_loss_final = 0.0
            self.v_loss_pose = 0.0
            self.v_loss_un = 0.0
        else:
            self.v_loss_final = sum_final / n_seen
            self.v_loss_pose = sum_pose / n_seen
            self.v_loss_un = sum_un / n_seen

        self.logger.info(
            f"[val] samples={n_seen}, "
            f"final={self.v_loss_final:.5f}, "
            f"pose={self.v_loss_pose:.5f}, "
            f"un={self.v_loss_un:.5f}")
