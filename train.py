"""
train.py -- Single-GPU training and evaluation script for MVOFormer.

Usage:
    # Training
    CUDA_VISIBLE_DEVICES=0 python train.py --mode train

    # Evaluation (requires GT poses)
    CUDA_VISIBLE_DEVICES=0 python train.py --mode eval

    # Evaluation with custom config and checkpoint epoch
    python train.py --mode eval --config Configs/MVOFormer.yaml --checkpoint 50
"""

import os
import argparse
import yaml
import datetime
import torch

from Tool.Train_Test.Trainer import Trainer
from Tool.Train_Test.Tester import Tester
from Tool.Utils.utils import set_random_seed, create_logger
from Tool.Datasets.multi_dataset import Path_set_multi
from Network.Model.model import build
from torch.utils.tensorboard import SummaryWriter


def deep_set(cfg, key, value):
    """Set nested config value from a dot-separated key path (e.g. 'model.is_Semantics')."""
    keys = key.split('.')
    for k in keys[:-1]:
        cfg = cfg[k]
    cfg[keys[-1]] = value


def parse_override(value):
    """Parse a string value into bool/int/float/None if possible."""
    if value.lower() in ('true', 'yes'):
        return True
    if value.lower() in ('false', 'no'):
        return False
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main(config, mode='train', checkpoint_epoch=None, overrides=None):
    assert os.path.exists(config)
    cfg = yaml.load(open(config, 'r'), Loader=yaml.Loader)

    # Apply command-line overrides
    if overrides:
        for kv in overrides:
            if '=' not in kv:
                continue
            key, value = kv.split('=', 1)
            value = parse_override(value)
            deep_set(cfg, key, value)

    # Single GPU device selection
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    set_random_seed(cfg.get('random_seed', 666))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(output_path, 'tensorboard_logs'))
    log_file = os.path.join(
        output_path,
        'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    logger = create_logger(log_file)
    logger.info(f"Single-GPU mode. device={device}")
    if torch.cuda.is_available():
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)}  "
            f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
        )

    # Multi-dataset path collection
    train_configs = cfg['dataset'].get('train_datasets', [])
    test_configs = cfg['dataset'].get('test_datasets', [])

    if not train_configs or not test_configs:
        raise ValueError(
            "Missing 'train_datasets' or 'test_datasets' in config. "
            "Please refer to MVOFormer.yaml for the expected format."
        )

    paths_dict = Path_set_multi(train_configs, test_configs)

    # Build model
    model, loss = build(cfg['model'])

    # Freeze DINOv3 backbone (inference always uses torch.no_grad)
    n_frozen_dinov3 = 0
    if hasattr(model, 'dinov3') and model.dinov3 is not None:
        for p in model.dinov3.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n_frozen_dinov3 += 1

    # Freeze delta_ref_point when pose_refine is disabled
    n_frozen_dref = 0
    if not cfg['model'].get('with_pose_refine', False):
        if hasattr(model, 'delta_ref_point'):
            for p in model.delta_ref_point.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    n_frozen_dref += 1
        if hasattr(model, 'fp_transformer') and \
           hasattr(model.fp_transformer, 'decoder') and \
           hasattr(model.fp_transformer.decoder, 'delta_ref_point') and \
           model.fp_transformer.decoder.delta_ref_point is not None:
            for p in model.fp_transformer.decoder.delta_ref_point.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    n_frozen_dref += 1

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[Param Freeze] dinov3: {n_frozen_dinov3} tensors frozen, "
        f"delta_ref_point: {n_frozen_dref} tensors frozen. "
        f"Trainable params: {n_train/1e6:.2f}M / {n_total/1e6:.2f}M"
    )

    model = model.to(device)

    try:
        if mode == 'eval':
            logger.info('################  Evaluation Only  ################')
            tester_cfg = cfg.copy()
            if checkpoint_epoch is not None:
                tester_cfg['tester']['checkpoint'] = checkpoint_epoch
            tester = Tester(cfg=tester_cfg, model=model, loss=loss, logger=logger,
                            paths_dict=paths_dict,
                            train_cfg=cfg['trainer'], model_name=model_name)
            tester.test()
            return

        if mode == 'train':
            trainer = Trainer(cfg=cfg, model=model, logger=logger, loss=loss,
                              paths_dict=paths_dict,
                              writer=writer, model_name=model_name)

            tester = Tester(cfg=cfg, model=model, loss=loss, logger=logger,
                            paths_dict=paths_dict,
                            train_cfg=cfg['trainer'], model_name=model_name)
            if cfg['dataset']['test_split'] != 'test':
                trainer.tester = tester

            logger.info('################  Training (Single GPU)  ################')
            logger.info('Batch Size: %d' % cfg['dataset']['batch_size'])
            logger.info('Learning Rate: %f' % cfg['optimizer']['lr'])
            logger.info('Num Queries: %d' % cfg['model']['num_queries'])

            trainer.train()
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVOFormer Training and Evaluation')
    parser.add_argument('--config', type=str, default='./Configs/MVOFormer.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Run mode: train or eval (evaluation with GT poses)')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Checkpoint epoch to evaluate (only for eval mode)')
    parser.add_argument('--set', action='append', dest='overrides', default=None,
                        help='Override config key=value (e.g. --set model.is_Semantics=False)')
    args = parser.parse_args()
    main(args.config, mode=args.mode, checkpoint_epoch=args.checkpoint, overrides=args.overrides)
