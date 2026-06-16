"""
infer.py -- Inference script for MVOFormer (no ground-truth poses required).

This script runs inference using pre-computed optical flow from the dataset.
It loads test datasets and model checkpoint path from the 'inference' section
of the config YAML.

Usage:
    # Single checkpoint inference
    python infer.py --config Configs/MVOFormer.yaml

    # Override checkpoint path
    python infer.py --config Configs/MVOFormer.yaml --checkpoint ./Model/MVOFormer.pth --mode single
"""

import os
import sys
import argparse
import yaml
import datetime
import torch

sys.path.append('./Network/SeaRAFT/core')
sys.path.append('./Network/SeaRAFT/config')

from Tool.Train_Test.Tester_img import Tester
from Tool.Utils.utils import set_random_seed, create_logger
from Tool.Datasets.multi_dataset import Path_set_multi
from Network.Model.model import build
from torch.utils.tensorboard import SummaryWriter


def main(config_path, mode=None, checkpoint_path=None, checkpoint_epoch=None):
    assert os.path.exists(config_path)
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 666))

    # ── inference section (new) ──
    inf_cfg = cfg.get('inference', {})

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(output_path, 'tensorboard_logs'))
    log_file = os.path.join(output_path,
                            'inference.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # ── Override config with inference section values ──
    # checkpoint: command-line > inference config > default
    ckpt_path = checkpoint_path or inf_cfg.get('checkpoint', None)
    ckpt_epoch = checkpoint_epoch if checkpoint_epoch is not None else inf_cfg.get('checkpoint_epoch', 50)
    inf_mode = mode or inf_cfg.get('mode', 'single')

    # Store direct checkpoint path into tester cfg so Tester_img can use it
    if ckpt_path:
        cfg['tester']['checkpoint_path'] = ckpt_path

    # Build model
    model, loss = build(cfg['model'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg['trainer'].get('gpu_ids', None):
        gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))
        if len(gpu_ids) == 1 or not torch.cuda.is_available():
            model = model.to(device)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)
    else:
        model = model.to(device)

    # Use inference datasets if available, otherwise fall back to test_datasets
    inf_datasets = inf_cfg.get('datasets', None)
    if inf_datasets:
        test_configs = inf_datasets
    else:
        test_configs = cfg['dataset'].get('test_datasets', [])

    logger.info('################  Inference (no GT required)  ################')
    logger.info('Num Queries: %d' % (cfg['model']['num_queries']))
    logger.info('Checkpoint: %s' % (ckpt_path or 'output dir'))

    paths_dict = Path_set_multi([], test_configs,
                                 train_split=cfg['dataset']['train_split'],
                                 test_split=cfg['dataset']['test_split'],
                                 require_flow=True)
    # Build a paths_dict with only test data (no train)
    paths_dict['train_img'] = []
    paths_dict['train_flow'] = []
    paths_dict['train_mask'] = []
    paths_dict['train_pose'] = []
    paths_dict['train_types'] = []
    paths_dict['train_repeats'] = {}

    # Override tester settings from inference config
    tester_cfg = cfg.copy()
    tester_cfg['tester']['checkpoint'] = ckpt_epoch
    tester_cfg['tester']['mode'] = inf_mode
    # Use inference batch_size / num_workers if specified
    if 'batch_size' in inf_cfg:
        tester_cfg['tester']['test_batch_size'] = inf_cfg['batch_size']
    if 'num_workers' in inf_cfg:
        tester_cfg['trainer']['num_workers'] = inf_cfg['num_workers']

    tester = Tester(cfg=tester_cfg,
                    model=model,
                    loss=loss,
                    logger=logger,
                    paths_dict=paths_dict,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVOFormer Inference')
    parser.add_argument('--config', type=str, default='./Configs/MVOFormer.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['single', 'all'],
                        help='Inference mode (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--checkpoint_epoch', type=int, default=None,
                        help='Checkpoint epoch number (overrides config)')
    args = parser.parse_args()
    main(args.config, mode=args.mode, checkpoint_path=args.checkpoint,
         checkpoint_epoch=args.checkpoint_epoch)
