"""
augmentation_sanity_check.py

Before training, run this script to visually verify:
  1. Random cropping produces views with different spatial positions (not always center).
  2. The intrinsic layer principal point shifts correctly with cropping.
  3. ColorJitter is applied synchronously to img1 and img2 (preserving photometric consistency).

Usage:
  CUDA_VISIBLE_DEVICES=0 python augmentation_sanity_check.py \\
      --config ./Configs/MVOFormer.yaml \\
      --ds_type tartanair_shibuya \\
      --num_views 12 \\
      --out ./Outputs/sanity_check

Output:
  ./Outputs/sanity_check/
    shibuya_view_00.png    -- 12 random augmentation views of the same image
    shibuya_view_01.png
    ...
    intrinsic_log.txt      -- (fx, fy, cx, cy) values for each view
"""
import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch

# Assume running from project root
sys.path.insert(0, os.path.abspath('.'))

from Tool.Datasets.VODataest import VODataset
from Tool.Datasets.multi_dataset import (
    Path_set_multi,
    build_train_transform,
    get_intrinsic_for,
)


def visualize_one_sample(sample, save_path, title=''):
    """Render img1, img2, flow, intrinsic as a 2x2 subplot grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # img1
    img1 = sample['img1'].numpy()
    if img1.max() > 2.0:
        img1 = img1 / 255.0
    img1 = np.clip(img1.transpose(1, 2, 0), 0, 1)
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('img1')
    axes[0, 0].axis('off')

    # img2
    img2 = sample['img2'].numpy()
    if img2.max() > 2.0:
        img2 = img2 / 255.0
    img2 = np.clip(img2.transpose(1, 2, 0), 0, 1)
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('img2')
    axes[0, 1].axis('off')

    # flow x component
    flow = sample['flow'].numpy()
    axes[1, 0].imshow(flow[0], cmap='RdBu_r')
    axes[1, 0].set_title(f'flow[x]  range=[{flow[0].min():.1f}, {flow[0].max():.1f}]')
    axes[1, 0].axis('off')

    # intrinsic x channel -- zero-crossing marks the principal point cx
    intr = sample['intrinsic'].numpy()
    # intrinsic channel is normalized coords (x-cx)/fx, so 0 marks cx
    im = axes[1, 1].imshow(intr[0], cmap='seismic',
                           vmin=-abs(intr[0]).max(), vmax=abs(intr[0]).max())
    axes[1, 1].set_title(
        f'intrinsic[x]  (cx at 0-value column, fx_eff≈{1.0/(intr[0, 0, 1]-intr[0, 0, 0]):.1f})'
    )
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=80, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./Configs/MVOFormer.yaml')
    parser.add_argument('--ds_type', default='tartanair_shibuya',
                        help='Dataset name to verify (must appear in train_datasets)')
    parser.add_argument('--num_views', type=int, default=12,
                        help='Number of random augmentation views to generate')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index into VODataset for repeated augmentation test')
    parser.add_argument('--out', default='./Outputs/sanity_check')
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    os.makedirs(args.out, exist_ok=True)

    # 1. scan paths
    train_cfgs = cfg['dataset']['train_datasets']
    test_cfgs  = cfg['dataset']['test_datasets']
    paths_dict = Path_set_multi(
        train_cfgs, test_cfgs,
        train_split=cfg['dataset'].get('train_split', 'train'),
        test_split=cfg['dataset'].get('test_split', 'test'),
    )

    # 2. filter paths for target ds_type
    imgs, flows, poses = [], [], []
    for i, t in enumerate(paths_dict['train_types']):
        if t == args.ds_type:
            imgs.append(paths_dict['train_img'][i])
            flows.append(paths_dict['train_flow'][i])
            poses.append(paths_dict['train_pose'][i])
    assert imgs, f"No samples of ds_type={args.ds_type} found"

    # 3. build transform and VODataset
    aug_cfg = cfg['dataset'].get('augmentation', None)
    target_size = tuple(cfg['dataset'].get('target_size', (480, 640)))
    transform = build_train_transform(
        args.ds_type, target_size=target_size, aug_cfg=aug_cfg,
    )
    ds = VODataset(imgs, flows, None, poses, transform=transform)
    print(f"[sanity_check] {args.ds_type} dataset total samples: {len(ds)}")

    # 4. repeatedly fetch sample_idx, observing N augmentation views
    log_lines = []
    fx_orig, fy_orig, cx_orig, cy_orig = get_intrinsic_for(args.ds_type)
    log_lines.append(
        f"Original intrinsics: fx={fx_orig:.2f}, fy={fy_orig:.2f}, "
        f"cx={cx_orig:.2f}, cy={cy_orig:.2f}"
    )
    log_lines.append("=" * 60)

    for k in range(args.num_views):
        sample = ds[args.sample_idx]  # same idx, different random aug each time

        # Recover effective fx, cx from intrinsic channel:
        #   intrinsic[0, i, j] = (j - cx + 0.5) / fx
        #   -> fx = 1 / (intrinsic[0, 0, 1] - intrinsic[0, 0, 0])
        #   -> cx = 0.5 - intrinsic[0, 0, 0] * fx
        intr = sample['intrinsic'].numpy()
        fx_eff = 1.0 / (intr[0, 0, 1] - intr[0, 0, 0])
        cx_eff = 0.5 - intr[0, 0, 0] * fx_eff
        fy_eff = 1.0 / (intr[1, 1, 0] - intr[1, 0, 0])
        cy_eff = 0.5 - intr[1, 0, 0] * fy_eff

        log_lines.append(
            f"view {k:02d}: "
            f"fx_eff={fx_eff:.2f} (x{fx_eff/fx_orig:.2f}), "
            f"cx_eff={cx_eff:.2f} (delta={cx_eff - target_size[1]/2:+.2f} from center), "
            f"cy_eff={cy_eff:.2f} (delta={cy_eff - target_size[0]/2:+.2f})"
        )

        save_path = os.path.join(
            args.out, f"{args.ds_type}_view_{k:02d}.png"
        )
        visualize_one_sample(
            sample, save_path,
            title=(f"{args.ds_type}  sample={args.sample_idx}  view={k}  |  "
                   f"fx={fx_eff:.0f}, cx={cx_eff:.1f}, cy={cy_eff:.1f}")
        )

    # 5. save numeric log
    log_path = os.path.join(args.out, 'intrinsic_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))

    print(f"[sanity_check] Generated {args.num_views} views in {args.out}")
    print(f"[sanity_check] See {log_path} for per-view intrinsic values")
    print("\nKey checks:")
    print("  - PNGs should show clearly different spatial crop positions (if keep_center=False)")
    print("  - cx_eff/cy_eff should be randomly distributed, not always at target_w/2=320, target_h/2=240")
    print("  - intrinsic[x] zero-crossing should shift randomly, not stay centered")
    print("  - img1 and img2 color shifts should be identical (ColorJitter synchronized)")


if __name__ == '__main__':
    main()
