import os
import re
import tqdm
import time
import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_start_method(method='forkserver', force=True)

from collections import defaultdict
from torch.utils.data import DataLoader
from Tool.Utils.load_save import load_checkpoint
from Tool.Utils.utils import my_worker_init_fn
from Tool.Datasets.transformation import motion_ses2pose_quats
from Tool.Datasets.VODataest import VODataset
from Tool.Datasets.utils import plot_traj
from Tool.Datasets.multi_dataset import (
    get_intrinsic_for,
    build_test_transform,
)
from Tool.Evaluator.tartanair_evaluator import TartanAirEvaluator


class Tester(object):
    """
    Multi-dataset tester. Each test sequence is assigned the appropriate
    intrinsic parameters based on its dataset type.

    Args:
        paths_dict: from Path_set_multi. Sequences are matched to intrinsics
                    via paths_dict['test_types'][i].
    """

    def __init__(self, cfg, model, loss, logger, paths_dict,
                 train_cfg=None, model_name='MVOFormer'):
        self.cfg = cfg['tester']
        self.model = model
        self.evaluator = TartanAirEvaluator()
        self.pose_loss = loss
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.epoch_num = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name
        self.paths_dict = paths_dict
        self.pose_std = np.array(cfg["dataset"]["pose_std"], dtype=np.float32)
        self.batch_size = cfg['tester']['test_batch_size']
        self.test_step = cfg['tester']['test_step']
        self.num_workers = cfg['trainer']['num_workers']
        self.scale = cfg['trainer']['scale']

    def extract_epoch_number(self, file_path):
        m = re.search(r'epoch_(\d+)', os.path.basename(file_path))
        return m.group(1) if m else None

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                ckpt = os.path.join(self.output_dir,
                                    "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                ckpt = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(ckpt)
            load_checkpoint(model=self.model, optimizer=None, filename=ckpt,
                            map_location=self.device, logger=self.logger)
            self.model.to(self.device)
            self.inference()

        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            pattern = r"_(\d+)\.pth"
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth"):
                        match = re.search(pattern, f)
                        if match and int(match.group(1)) >= start_epoch:
                            checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for ckpt in checkpoints_list:
                load_checkpoint(model=self.model, optimizer=None, filename=ckpt,
                                map_location=self.device, logger=self.logger)
                self.epoch_num = self.extract_epoch_number(ckpt)
                self.model.to(self.device)
                self.inference()

    @staticmethod
    def _make_seq_name(ds_type, pose_path):
        """
        Generate a unique sequence visualization name: {ds_type}__{scene}__{filename}.
        Example: pose_path = ".../KITTI/00/test_pose/pose_left"
        -> ds_type='kitti', scene='00', filename='pose_left'.
        """
        base = os.path.basename(pose_path)
        filename = os.path.splitext(base)[0] if '.' in base else base
        scene_dir = os.path.basename(os.path.dirname(os.path.dirname(pose_path))) \
                    if pose_path else 'unknown'
        if not scene_dir or scene_dir == 'unknown':
            return f"{ds_type}__{filename}"
        return f"{ds_type}__{scene_dir}__{filename}"

    def inference(self, is_plot_traj=True):
        torch.set_grad_enabled(False)
        self.model.eval()

        img_path = self.paths_dict['test_img']
        flows_path = self.paths_dict['test_flow']
        poses_path = self.paths_dict['test_pose']
        types_path = self.paths_dict['test_types']

        assert len(img_path) == len(poses_path) == len(types_path), (
            f"Length mismatch: imgs={len(img_path)}, poses={len(poses_path)}, "
            f"types={len(types_path)}")
        if len(flows_path) != len(poses_path):
            self.logger.warning(
                f"[Tester] flow list length ({len(flows_path)}) != pose "
                f"({len(poses_path)}) -- some sequences may fail when accessing flow")

        progress_bar = tqdm.tqdm(total=len(poses_path), leave=True,
                                 desc='Evaluation Progress',
                                 dynamic_ncols=True, position=0,
                                 mininterval=0.1, smoothing=0.1)

        model_infer_time = 0
        ate_score_allTest = 0
        results_dict = {}

        # Cache test transforms per dataset type to avoid repeated construction
        transform_cache = {}

        for len_data in range(len(poses_path)):
            test_imgs = [img_path[len_data]]
            test_flows = [flows_path[len_data]] if len_data < len(flows_path) else [None]
            test_poses = [poses_path[len_data]]
            ds_type = types_path[len_data]
            motionlist = []

            if ds_type not in transform_cache:
                transform_cache[ds_type] = build_test_transform(ds_type,
                                                                target_size=(480, 640))
                fx, fy, cx, cy = get_intrinsic_for(ds_type)

                # Compute effective principal point after CropCenter correction
                _orig_wh = {
                    'tartanair': (640, 480),
                    'tartanair_shibuya': (640, 360),
                    'kitti': (1241, 376),
                    'euroc': (752, 480),
                    'tum': (640, 480),
                }
                th_t, tw_t = 480, 640
                if ds_type in _orig_wh:
                    w_o, h_o = _orig_wh[ds_type]
                    s = max(1.0, th_t / float(h_o), tw_t / float(w_o))
                    w_s = int(round(w_o * s))
                    h_s = int(round(h_o * s))
                    x1 = int((w_s - tw_t) / 2)
                    y1 = int((h_s - th_t) / 2)
                    cx_eff = cx * s - x1
                    cy_eff = cy * s - y1
                    dcx = cx_eff - tw_t / 2.0
                    dcy = cy_eff - th_t / 2.0
                    if abs(dcx) > 1.0 or abs(dcy) > 1.0:
                        fix_note = (
                            f"\n    [Principal Point Correction] original {w_o}x{h_o}, "
                            f"scale={s:.3f} -> corrected cx_eff={cx_eff:.2f} "
                            f"(was {tw_t/2:.0f}, delta={dcx:+.2f}), "
                            f"cy_eff={cy_eff:.2f} (was {th_t/2:.0f}, delta={dcy:+.2f})")
                    else:
                        fix_note = ("  [Principal point near image center -- "
                                    f"delta=({dcx:+.2f}, {dcy:+.2f}) pixels]")
                else:
                    fix_note = ("  [Original image size not registered -- "
                                "CropCenter will print actual correction on first call]")

                self.logger.info(
                    f"[Tester] new dataset '{ds_type}': "
                    f"original intrinsic (fx={fx:.1f}, fy={fy:.1f}, "
                    f"cx={cx:.1f}, cy={cy:.1f}){fix_note}")
            transform_test = transform_cache[ds_type]

            test_Dataset = VODataset(test_imgs, test_flows, None, test_poses,
                                     transform=transform_test, is_test=True)
            dataloader = DataLoader(test_Dataset, batch_size=self.batch_size,
                                    prefetch_factor=2, shuffle=False,
                                    num_workers=self.num_workers,
                                    worker_init_fn=my_worker_init_fn,
                                    drop_last=False, persistent_workers=True)

            seq_viz_name = self._make_seq_name(ds_type, test_poses[0])

            for batch_idx, data in enumerate(dataloader):
                img1 = data['img1'].cuda(non_blocking=True)
                img2 = data['img2'].cuda(non_blocking=True)
                flow = data['flow'].cuda(non_blocking=True)
                pose = data['pose'].cuda(non_blocking=True)
                intrinsic = data['intrinsic'].cuda(non_blocking=True)
                inputs = [img1, img2, flow, intrinsic]

                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(inputs)
                end_time = time.time()
                model_infer_time += end_time - start_time

                Translations = outputs['outputs_pose_translations']
                Rots = outputs['outputs_pose_rots']
                Pose = torch.cat((Translations, Rots), dim=1)

                posenp = Pose.data.cpu().numpy() * self.pose_std
                motions_gt = pose.cpu()
                scale = np.linalg.norm(motions_gt[:, :3], axis=1)
                trans_est = posenp[:, :3]
                trans_est = trans_est / np.linalg.norm(trans_est, axis=1).reshape(-1, 1) \
                            * scale.reshape(-1, 1)
                posenp[:, :3] = trans_est
                motionlist.extend(posenp)

            estposes = motion_ses2pose_quats(np.array(motionlist))
            evaluator = TartanAirEvaluator()
            results = evaluator.evaluate_one_trajectory(test_poses, estposes,
                                                        scale=True, kittitype=True)

            if is_plot_traj:
                seq_path = self.output_dir + '_results_' + str(self.epoch_num) + '/'
                os.makedirs(seq_path, exist_ok=True)
                seq_name = seq_path + 'test_' + seq_viz_name + '.png'
                plot_traj(results['gt_aligned'], results['est_aligned'],
                          savefigname=seq_name,
                          title='ATE %.4f' % (results['ate_score']))
                np.savetxt(seq_name + '.txt', results['est_aligned'])

            results_dict[seq_viz_name] = {
                "dataset": ds_type,
                "ATE": results['ate_score'],
                "inference_time": model_infer_time / max(1, (len(dataloader) / self.batch_size)),
                "num_images": len(dataloader),
                "ATE_scale": results['ATE_scale'],
                "kitti_score": results['kitti_score'],
            }
            ate_score_allTest += results['ate_score']

            progress_bar.update()
            progress_bar.refresh()
        progress_bar.close()

        # Print results grouped by dataset
        by_dataset = defaultdict(list)
        for key, value in results_dict.items():
            by_dataset[value['dataset']].append((key, value))

        for ds_type, entries in by_dataset.items():
            print(f"\n===== Dataset: {ds_type}  ({len(entries)} sequences) =====")
            for key, value in entries:
                print(f"  {key}: ATE={value['ATE']:.4f}/Scale:{value['ATE_scale']}, "
                      f"kitti_score:{value['kitti_score']}, "
                      f"Time/Image={value['inference_time']:.4f}s ({value['num_images']} images)")
            ds_mean = np.mean([v['ATE'] for _, v in entries])
            print(f"  --- {ds_type} mean ATE: {ds_mean:.5f} ---")

        ate_score_allTest_ave = ate_score_allTest / max(1, len(poses_path))
        print("\n----- Total ATE Average (all datasets): %.5f -----" % ate_score_allTest_ave)
        self.logger.info("Result test_ate_loss:{}".format(ate_score_allTest_ave))

        return ate_score_allTest_ave
