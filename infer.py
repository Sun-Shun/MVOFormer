"""
infer.py -- Inference script for MVOFormer.

Usage:
    # Dataset inference (requires pre-computed flow in dataset folder)
    python infer.py --config Configs/MVOFormer.yaml --checkpoint ./Model/MVOFormer.pth --mode single

    # Raw image folder inference (on-the-fly flow computation)
    python infer.py --img_folder /path/to/images --checkpoint ./Model/MVOFormer.pth

    # Video inference
    python infer.py --video /path/to/video.mp4 --checkpoint ./Model/MVOFormer.pth

    # Inference options
    python infer.py --video /path/to/video.mp4 --checkpoint ./Model/MVOFormer.pth --fx 320 --fy 320 --cx 320 --cy 240
"""

import os
import sys
import argparse
import yaml
import datetime
import re
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, './Network/SeaRAFT')
sys.path.insert(0, './Network/SeaRAFT/core')
sys.path.insert(0, './Network/SeaRAFT/config')

from Tool.Train_Test.Tester_img import Tester
from Tool.Utils.utils import set_random_seed, create_logger
from Tool.Datasets.multi_dataset import Path_set_multi
from Tool.Datasets.transformation import motion_ses2pose_quats
from Tool.Datasets.utils import make_intrinsics_layer, plot_traj
from Network.Model.model import build
from raft import RAFT
from torch.utils.tensorboard import SummaryWriter

from config.parser import json_to_args


def load_sea_raft(model_path, config_path, device):
    args = json_to_args(config_path)
    model = RAFT(args)
    state = torch.load(model_path, map_location='cuda', weights_only=True)
    if 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model, args


@torch.no_grad()
def compute_flow(sea_raft, args, img1, img2, device):
    """Compute optical flow between two normalized images [1,3,H,W] on device."""
    s = 2 ** args.scale
    img1_in = F.interpolate(img1, scale_factor=s, mode='bilinear', align_corners=False)
    img2_in = F.interpolate(img2, scale_factor=s, mode='bilinear', align_corners=False)
    output = sea_raft(img1_in, img2_in, iters=args.iters, test_mode=True)
    flow = output['flow'][-1]
    flow = F.interpolate(flow, scale_factor=1.0 / s, mode='bilinear', align_corners=False) * (1.0 / s)
    return flow


def infer_raw(images, cfg, model, loss, logger, output_path, model_name, pose_std,
              sea_raft=None, sea_raft_args=None, scale=4, intrinsic=None):
    """
    Run inference on a list of raw images (RGB numpy uint8 [H,W,3]).
    If sea_raft is provided, flow is computed on the fly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # ── Warm up ──
    logger.info("Warming up model ...")
    with torch.no_grad():
        dummy = torch.randn(1, 3, 480, 640, device=device)
        dummy_f = torch.randn(1, 2, 480, 640, device=device)
        dummy_i = torch.randn(1, 2, 480, 640, device=device)
        _ = model([dummy, dummy, dummy_f, dummy_i])
    torch.cuda.synchronize()

    H, W = images[0].shape[:2]
    # If intrinsic not provided, use default TartanAir intrinsics
    if intrinsic is None:
        fx = fy = 320.0
        cx, cy = W / 2.0, H / 2.0
    else:
        fx, fy, cx, cy = intrinsic
    intrinsic_layer = make_intrinsics_layer(W, H, fx, fy, cx, cy).to(device)

    # ── Inference ──
    motionlist = []
    total_time = 0.0
    n_frames = 0

    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    if m.dinov3 is not None:
        m.dinov3.reset_rnn_state()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    logger.info(f"Processing {len(images)} images ({H}x{W}) ...")

    for i in range(len(images) - 1):
        # Load & preprocess images
        img1_np = images[i].astype(np.float32) / 255.0
        img2_np = images[i + 1].astype(np.float32) / 255.0
        img1 = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0).to(device)
        img2 = torch.from_numpy(img2_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Compute flow
        with torch.no_grad():
            flow = compute_flow(sea_raft, sea_raft_args, img1, img2, device)

        # Resize to target (480, 640) if needed
        if H != 480 or W != 640:
            img1 = F.interpolate(img1, size=(480, 640), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(480, 640), mode='bilinear', align_corners=False)
            flow = F.interpolate(flow, size=(480, 640), mode='bilinear', align_corners=False)

        inputs = [img1, img2, flow, intrinsic_layer.unsqueeze(0)]

        starter.record()
        with torch.no_grad():
            outputs = model(inputs, rnn_time=True)
        ender.record()
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender)
        total_time += elapsed
        n_frames += 1

        t = outputs['outputs_pose_translations']
        r = outputs['outputs_pose_rots']
        pose = torch.cat((t, r), dim=1).cpu().numpy() * pose_std
        motionlist.append(pose[0])

    # ── Convert to trajectory ──
    estposes = motion_ses2pose_quats(np.array(motionlist))
    seq_path = os.path.join(output_path, f'{model_name}_results')
    os.makedirs(seq_path, exist_ok=True)
    traj_path = os.path.join(seq_path, 'trajectory')
    plot_traj(None, estposes, savefigname=traj_path + '.png', title='Estimated Trajectory')
    np.savetxt(traj_path + '.txt', estposes)

    avg_ms = total_time / max(n_frames, 1)
    logger.info(f"\n{'='*50}")
    logger.info(f"Frames processed : {n_frames}")
    logger.info(f"Total time       : {total_time/1000:.2f}s")
    logger.info(f"Avg per frame    : {avg_ms:.2f}ms")
    logger.info(f"FPS              : {1000.0/avg_ms:.1f}")
    logger.info(f"Trajectory saved : {traj_path}.png / .txt")
    logger.info(f"{'='*50}")


def load_images_from_folder(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f'*{ext}')))
        paths.extend(glob.glob(os.path.join(folder, f'*{ext.upper()}')))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images


def extract_frames_from_video(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")
    return frames


def main(config_path, mode=None, checkpoint_path=None, checkpoint_epoch=None,
         img_folder=None, video_path=None, fx=None, fy=None, cx=None, cy=None):
    assert os.path.exists(config_path), f"Config not found: {config_path}"
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 666))

    inf_cfg = cfg.get('inference', {})
    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path,
                            'inference.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    ckpt_path = checkpoint_path or inf_cfg.get('checkpoint', None)
    if ckpt_path is None:
        ckpt_epoch = checkpoint_epoch if checkpoint_epoch is not None else inf_cfg.get('checkpoint_epoch', 50)
        ckpt_path = os.path.join(output_path, f"checkpoint_epoch_{ckpt_epoch}.pth")
    logger.info(f"Checkpoint: {ckpt_path}")

    # Build model
    model, loss = build(cfg['model'])
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    logger.info("Model loaded")

    # ── Raw image / video inference ──
    if img_folder or video_path:
        sea_raft_path = cfg['model'].get('SeaRAFT_model', None)
        assert sea_raft_path and os.path.exists(sea_raft_path), \
            f"SeaRAFT model not found: {sea_raft_path}. Set model.SeaRAFT_model in config."
        # Find the corresponding config JSON
        raft_configs = glob.glob('Network/SeaRAFT/config/train/*.json')
        raft_config_path = None
        for rc in raft_configs:
            if 'Tartan-C-T-TSKH-kitti432x960-S' in rc:
                raft_config_path = rc
                break
        if raft_config_path is None and raft_configs:
            raft_config_path = raft_configs[0]
        assert raft_config_path, "No SeaRAFT config JSON found in Network/SeaRAFT/config/train/"
        logger.info(f"SeaRAFT config: {raft_config_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sea_raft, raft_args = load_sea_raft(sea_raft_path, raft_config_path, device)
        logger.info("SeaRAFT model loaded")

        pose_std = np.array(cfg["dataset"]["pose_std"], dtype=np.float32)
        intrinsic = (fx, fy, cx, cy) if fx is not None else None

        if img_folder:
            logger.info(f"Loading images from: {img_folder}")
            images = load_images_from_folder(img_folder)
        else:
            logger.info(f"Extracting frames from: {video_path}")
            images = extract_frames_from_video(video_path)

        logger.info(f"Total frames: {len(images)}")
        infer_raw(images, cfg, model, loss, logger, output_path, model_name,
                  pose_std, sea_raft, raft_args, scale=cfg['trainer']['scale'],
                  intrinsic=intrinsic)

    # ── Dataset inference (original) ──
    else:
        writer = SummaryWriter(log_dir=os.path.join(output_path, 'tensorboard_logs'))
        ckpt_epoch = checkpoint_epoch if checkpoint_epoch is not None else inf_cfg.get('checkpoint_epoch', 50)
        inf_mode = mode or inf_cfg.get('mode', 'single')

        if ckpt_path:
            cfg['tester']['checkpoint_path'] = ckpt_path

        cfg['tester']['checkpoint'] = ckpt_epoch
        cfg['tester']['mode'] = inf_mode

        inf_datasets = inf_cfg.get('datasets', None)
        if inf_datasets:
            test_configs = inf_datasets
        else:
            test_configs = cfg['dataset'].get('test_datasets', [])

        logger.info('################  Dataset Inference  ################')
        logger.info('Num Queries: %d' % (cfg['model']['num_queries']))
        logger.info('Checkpoint: %s' % (ckpt_path or 'output dir'))

        paths_dict = Path_set_multi([], test_configs,
                                     train_split=cfg['dataset']['train_split'],
                                     test_split=cfg['dataset']['test_split'],
                                     require_flow=True)
        paths_dict['train_img'] = []
        paths_dict['train_flow'] = []
        paths_dict['train_mask'] = []
        paths_dict['train_pose'] = []
        paths_dict['train_types'] = []
        paths_dict['train_repeats'] = {}

        tester_cfg = cfg.copy()
        if 'batch_size' in inf_cfg:
            tester_cfg['tester']['test_batch_size'] = inf_cfg['batch_size']
        if 'num_workers' in inf_cfg:
            tester_cfg['trainer']['num_workers'] = inf_cfg['num_workers']

        tester = Tester(cfg=tester_cfg, model=model, loss=loss, logger=logger,
                        paths_dict=paths_dict, train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()

        if writer is not None:
            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVOFormer Inference')
    parser.add_argument('--config', type=str, default='./Configs/MVOFormer.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['single', 'all'],
                        help='Inference mode for dataset (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--checkpoint_epoch', type=int, default=None,
                        help='Checkpoint epoch number (overrides config)')

    # Raw input options
    parser.add_argument('--img_folder', type=str, default=None,
                        help='Path to folder with images for direct inference')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file for direct inference')
    parser.add_argument('--fx', type=float, default=None,
                        help='Camera fx (default: auto from dataset type)')
    parser.add_argument('--fy', type=float, default=None,
                        help='Camera fy (default: auto from dataset type)')
    parser.add_argument('--cx', type=float, default=None,
                        help='Camera cx (default: image center)')
    parser.add_argument('--cy', type=float, default=None,
                        help='Camera cy (default: image center)')

    args = parser.parse_args()
    main(args.config, mode=args.mode, checkpoint_path=args.checkpoint,
         checkpoint_epoch=args.checkpoint_epoch,
         img_folder=args.img_folder, video_path=args.video,
         fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
