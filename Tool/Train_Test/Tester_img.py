import os
import re
import tqdm
import time
import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_start_method(method='forkserver', force=True)

from torch.utils.data import DataLoader
from Tool.Utils.load_save import load_checkpoint
from Tool.Utils.utils import my_worker_init_fn
from Tool.Datasets.transformation import motion_ses2pose_quats
from Tool.Datasets.VODataest import VODataset
from Tool.Datasets.utils import plot_traj
from Tool.Datasets.multi_dataset import get_intrinsic_for, build_test_transform
from Tool.Evaluator.tartanair_evaluator import TartanAirEvaluator


class Tester(object):
    """
    Inference tester. Loads pre-computed optical flow from the dataset and runs
    the model without requiring ground-truth poses.

    Model input: [img1, img2, flow, intrinsic]  (4 elements).
    """

    def __init__(self, cfg, model, loss, logger, paths_dict, train_cfg=None,
                 model_name='MVOFormer'):
        self.cfg = cfg['tester']
        self.model = model
        self.evaluator = TartanAirEvaluator()
        self.pose_loss = loss
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg['dataset']['type']
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
        self.intrinsic = get_intrinsic_for(self.dataset_type)
        self.transform_test = build_test_transform(self.dataset_type,
                                                    target_size=(480, 640))

    def extract_epoch_number(self, file_path):
        file_name = os.path.basename(file_path)
        match = re.search(r'epoch_(\d+)', file_name)
        return match.group(1) if match else None

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # Direct checkpoint path (from inference config)
        ckpt_path = self.cfg.get('checkpoint_path', None)

        if ckpt_path:
            assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
            load_checkpoint(model=self.model, optimizer=None, filename=ckpt_path,
                            map_location=self.device, logger=self.logger)
            self.model.to(self.device)
            self.inference()

        elif self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                ckpt = os.path.join(
                    self.output_dir,
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

    def inference(self, is_plot_traj=True):
        torch.set_grad_enabled(False)
        self.model.eval()

        test_img = self.paths_dict['test_img']
        test_flow = self.paths_dict['test_flow']
        test_pose = self.paths_dict['test_pose']

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        total_frames = 0
        total_inference_time_ms = 0.0

        # Warm-up
        self.logger.info("Warming up model ...")
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 480, 640, device=self.device)
            dummy_flow = torch.randn(1, 2, 480, 640, device=self.device)
            dummy_intrinsic = torch.randn(1, 2, 480, 640, device=self.device)
            _ = self.model([dummy_img, dummy_img, dummy_flow, dummy_intrinsic])
        torch.cuda.synchronize()

        progress_bar = tqdm.tqdm(total=len(test_pose), leave=True,
                                 desc='Evaluation', dynamic_ncols=True,
                                 position=0, mininterval=0.1, smoothing=0.1)

        for len_data in range(len(test_pose)):
            # Reset DINOv3 RNN state for each new sequence
            m = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            if m.dinov3 is not None:
                m.dinov3.reset_rnn_state()

            test_imgs = [test_img[len_data]]
            test_flows = [test_flow[len_data]]
            test_poses = [test_pose[len_data]]
            motionlist = []

            test_Dataset = VODataset(test_imgs, test_flows, None, test_poses,
                                     transform=self.transform_test, is_test=True)
            dataloader = DataLoader(test_Dataset, batch_size=self.batch_size,
                                    prefetch_factor=2, shuffle=False,
                                    num_workers=self.num_workers,
                                    worker_init_fn=my_worker_init_fn,
                                    drop_last=False, persistent_workers=True)

            file_name_without_ext = os.path.splitext(os.path.basename(test_poses[0]))[0]

            for batch_idx, data in enumerate(dataloader):
                img1 = data['img1'].cuda(non_blocking=True)
                img2 = data['img2'].cuda(non_blocking=True)
                flow = data['flow'].cuda(non_blocking=True)
                intrinsic = data['intrinsic'].cuda(non_blocking=True)
                inputs = [img1, img2, flow, intrinsic]

                starter.record()
                outputs = self.model(inputs, rnn_time=True)
                ender.record()
                torch.cuda.synchronize()
                elapsed_ms = starter.elapsed_time(ender)

                total_inference_time_ms += elapsed_ms
                total_frames += img1.shape[0]

                Translations = outputs['outputs_pose_translations']
                Rots = outputs['outputs_pose_rots']
                Pose = torch.cat((Translations, Rots), dim=1)
                posenp = Pose.cpu().numpy() * self.pose_std
                motionlist.extend(posenp)

            estposes = motion_ses2pose_quats(np.array(motionlist))
            if is_plot_traj:
                seq_path = os.path.join(self.output_dir + '_results_' + str(self.epoch_num))
                os.makedirs(seq_path, exist_ok=True)
                seq_name = os.path.join(seq_path, f'test_{file_name_without_ext}')
                plot_traj(None, estposes, savefigname=seq_name + '.png',
                         title='Estimated Trajectory')
                np.savetxt(seq_name + '.txt', estposes)

            progress_bar.update()
            progress_bar.refresh()

        progress_bar.close()

        self._print_speed_report(total_frames, total_inference_time_ms, len(test_pose))

    @staticmethod
    def _print_speed_report(total_frames, total_ms, n_sequences):
        if total_frames <= 0:
            print("Warning: no frames processed.")
            return
        avg_ms = total_ms / total_frames
        total_sec = total_ms / 1000.0
        bar = "=" * 70
        print("\n" + bar)
        print(" " * 20 + "INFERENCE SPEED REPORT")
        print(bar)
        print(f"Total sequences      : {n_sequences}")
        print(f"Total frames         : {total_frames}")
        print(f"Total inference time : {total_sec:.2f} s")
        print(f"Average per frame    : {avg_ms:.2f} ms")
        print(f"Real-time FPS        : {1000.0 / avg_ms:.1f}")
        print(bar)
