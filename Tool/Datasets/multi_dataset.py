"""
Multi-dataset support for VO training.

Option C (native intrinsics) + weighted oversampling (repeat) + spatial random crop augmentation.

Changes from the previous version:

1. [New] RandomResizeCropV2
     Fixes principal point tracking error when keep_center=False.
     The old RandomResizeCrop was only accidentally correct when keep_center=True
     and cx=w/2, cy=h/2. When spatial random cropping is enabled, the new version
     is required, otherwise the intrinsic channel will be corrupted.
     The old RandomResizeCrop in utils.py is kept unchanged for backward compat.

2. [New] AugmentedRepeatDataset
     Implements oversampling via "virtual length N x original length" instead of
     stuffing the same VODataset object into ConcatDataset multiple times.
     Functionally equivalent, but cleaner: each __getitem__ runs the full transform,
     so random cropping naturally yields different views. Friendlier for DataLoader
     shuffling.

3. [New] Augmentation config entry point
     Passed via build_multi_train_dataset(..., aug_cfg=...), for example:
         aug_cfg = {
             'keep_center': False,
             'max_scale': 2.5,
             'color_jitter': {'brightness': 0.2, 'contrast': 0.2,
                              'saturation': 0.1},
         }

Three main entry points are unchanged:
    1. Path_set_multi(train_cfgs, test_cfgs, train_split, test_split)
    2. build_multi_train_dataset(paths_dict, ...)
    3. get_intrinsic_for(dataset_type)
"""

import math
import random
import numbers
from glob import glob
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset

from Tool.Datasets.VODataest import VODataset
from Tool.Datasets.utils import (
    Compose, CropCenter, RandomResizeCrop,
    make_intrinsics_layer, generate_random_scale_crop,
)


# =============================================================================
# 1. Known dataset intrinsics registry (fx, fy, cx, cy in pixel units)
# =============================================================================
DATASET_INTRINSICS = {
    'tartanair':          (320.0,              320.0,              320.0,   240.0),
    'tartanair_shibuya':  (772.5483399593904,  772.5483399593904,  320.0,   180.0),
    'kitti':              (707.0912,           707.0912,           601.8873, 183.1104),
    'euroc':              (458.654,            457.296,            367.215, 248.375),
    'tum':                (517.3,              516.5,              318.6,   255.3),
    'bonn':               (517.3,              516.5,              318.6,   255.3),
    'ETH3D':              (726.21081542969,  726.21081542969,  359.2048034668, 202.47247314453),
}


def get_intrinsic_for(dataset_type):
    if dataset_type not in DATASET_INTRINSICS:
        raise KeyError(
            f"Unknown dataset '{dataset_type}'. "
            f"Known: {list(DATASET_INTRINSICS.keys())}. "
            f"Please add it to DATASET_INTRINSICS in Tool/Datasets/multi_dataset.py."
        )
    return DATASET_INTRINSICS[dataset_type]


# =============================================================================
# 2. RandomResizeCropV2 -- new crop transform with correct principal point tracking
# =============================================================================
# The old RandomResizeCrop always used image center for the intrinsic layer:
#     sample['intrinsic'] = make_intrinsics_layer(
#         target_w, target_h, fx*scale, fy*scale,
#         target_w/2.0, target_h/2.0,  # <-- always image center, WRONG!
#     )
# This was only accidentally correct when keep_center=True AND cx=w/2.
#
# This class correctly tracks the principal point through every crop/scale step,
# so it works correctly even when keep_center=False.
# =============================================================================
class RandomResizeCropV2(object):
    """
    Random scale + crop with CORRECT principal-point tracking.

    Pipeline:
        Step 1 (scale):   (w, h) -> (n_w, n_h) = (w*scale_s_w, h*scale_s_h)
        Step 2 (crop):    take (crop_w, crop_h) window at position (x1, y1)
        Step 3 (scale_2): if crop is smaller than target, upscale
        Step 4 (center crop): final center crop to (target_w, target_h)

    Principal point transform (tracked at every step):
        cx_1 = cx * scale_s_w
        cx_2 = cx_1 - x1
        cx_3 = cx_2 * scale_2
        cx_final = cx_3 - x1_2
    """

    def __init__(self, size, intrinsic, max_scale=2.5,
                 keep_center=False, fix_ratio=False, use_gpu=False):
        if isinstance(size, numbers.Number):
            self.target_h = int(size)
            self.target_w = int(size)
        else:
            self.target_h = size[0]
            self.target_w = size[1]

        self.keep_center = keep_center
        self.fix_ratio = fix_ratio
        self.fx = intrinsic[0]
        self.fy = intrinsic[1]
        self.cx = intrinsic[2]
        self.cy = intrinsic[3]
        self.use_gpu = use_gpu
        self.scale_base = max_scale

    def __call__(self, sample):
        device = torch.device('cuda' if torch.cuda.is_available()
                              and self.use_gpu else 'cpu')

        for kk in sample:
            if isinstance(sample[kk], torch.Tensor):
                sample[kk] = sample[kk].to(device)

        # Get original H, W
        for kk in sample:
            if isinstance(sample[kk], torch.Tensor) and sample[kk].dim() >= 2:
                h, w = sample[kk].shape[1], sample[kk].shape[2]
                break

        # Step 1 + 2: random scale + random position crop
        scale_w, scale_h, x1, y1, crop_w, crop_h = generate_random_scale_crop(
            h, w, self.scale_base, self.keep_center, self.fix_ratio
        )

        # Step 3: if target > crop, need to upscale
        scale_h_2, scale_w_2, scale_2 = 1., 1., 1.
        if self.target_h > crop_h:
            scale_h_2 = float(self.target_h) / crop_h
        if self.target_w > crop_w:
            scale_w_2 = float(self.target_w) / crop_w
        if scale_h_2 > 1 or scale_w_2 > 1:
            scale_2 = max(scale_h_2, scale_w_2)
            crop_w_2 = int(round(crop_w * scale_2))
            crop_h_2 = int(round(crop_h * scale_2))
            x1_2 = int((crop_w_2 - self.target_w) / 2)
            y1_2 = int((crop_h_2 - self.target_h) / 2)
        else:
            crop_w_2 = crop_w
            crop_h_2 = crop_h
            x1_2 = int((crop_w - self.target_w) / 2)
            y1_2 = int((crop_h - self.target_h) / 2)

        # Resize + crop for each spatial channel
        for nn in sample:
            if nn == 'pose' or not isinstance(sample[nn], torch.Tensor):
                continue
            sample[nn] = F.interpolate(
                sample[nn].unsqueeze(0),
                scale_factor=(scale_h, scale_w),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            sample[nn] = sample[nn][:, y1:y1 + crop_h, x1:x1 + crop_w]
            if scale_2 > 1.0:
                sample[nn] = F.interpolate(
                    sample[nn].unsqueeze(0),
                    size=(crop_h_2, crop_w_2),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            sample[nn] = sample[nn][:, y1_2:y1_2 + self.target_h,
                                       x1_2:x1_2 + self.target_w]

        # Safety net: force all spatial tensors to exact target_size
        for nn in sample:
            if nn == 'pose' or not isinstance(sample[nn], torch.Tensor):
                continue
            t = sample[nn]
            if t.ndim == 3 and (t.shape[1] != self.target_h or t.shape[2] != self.target_w):
                # Occasional off-by-one, force-align via interpolate
                t = F.interpolate(
                    t.unsqueeze(0).float(),     # interpolate doesn't accept uint8
                    size=(self.target_h, self.target_w),
                    mode='bilinear', align_corners=False,
                ).squeeze(0).to(sample[nn].dtype)
                sample[nn] = t

        # Core: correct principal point tracking
        fx_final = self.fx * scale_w * scale_2
        fy_final = self.fy * scale_h * scale_2
        cx_final = (self.cx * scale_w - x1) * scale_2 - x1_2
        cy_final = (self.cy * scale_h - y1) * scale_2 - y1_2

        sample['intrinsic'] = make_intrinsics_layer(
            self.target_w, self.target_h,
            fx_final, fy_final, cx_final, cy_final,
            use_gpu=self.use_gpu,
        ).to(device)

        # Flow needs scaling by the spatial scale factors
        if 'flow' in sample:
            sample['flow'][0, :, :] = sample['flow'][0, :, :] * scale_w * scale_2
            sample['flow'][1, :, :] = sample['flow'][1, :, :] * scale_h * scale_2

        # Return to CPU so pin_memory works correctly
        if self.use_gpu:
            for kk in sample:
                if isinstance(sample[kk], torch.Tensor):
                    sample[kk] = sample[kk].cpu()

        return sample


# =============================================================================
# 3. ColorJitter -- optional photometric augmentation (synchronized on img1/img2)
# =============================================================================
class ColorJitter(object):
    """
    Apply the same brightness/contrast/saturation perturbation to img1 and img2.

    Purpose: help the backbone (DINOv3/ResNet) learn illumination invariance,
    mitigating the photometric distribution gap between shibuya (real daylight)
    and tartanair (synthetic rendering).

    Notes:
        - Same perturbation parameters for both img1 and img2 (preserving
          inter-frame consistency).
        - Does not touch flow / pose / intrinsic / mask (geometry channels are
          independent of photometry).
        - Assumes input images are in [0, 255] range (CV reads uint8 -> float).
    """

    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    @staticmethod
    def _rand_factor(delta):
        """Return a random multiplier in [1-delta, 1+delta]."""
        if delta <= 0:
            return 1.0
        return 1.0 + (random.random() * 2 - 1) * delta

    def __call__(self, sample):
        if 'img1' not in sample:
            return sample

        b = self._rand_factor(self.brightness)
        c = self._rand_factor(self.contrast)
        s = self._rand_factor(self.saturation)

        for key in ['img1', 'img2']:
            if key not in sample or not isinstance(sample[key], torch.Tensor):
                continue
            img = sample[key]

            # Brightness: overall scale
            if self.brightness > 0:
                img = img * b

            # Contrast: scale around mean
            if self.contrast > 0:
                mean = img.mean(dim=(1, 2), keepdim=True)
                img = (img - mean) * c + mean

            # Saturation: scale around grayscale (RGB channels)
            if self.saturation > 0 and img.shape[0] == 3:
                gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])
                gray = gray.unsqueeze(0).expand_as(img)
                img = (img - gray) * s + gray

            # Clamp range
            img = img.clamp(0.0, 255.0)
            sample[key] = img

        return sample


# =============================================================================
# 4. AugmentedRepeatDataset -- virtual oversampling wrapper
# =============================================================================
class AugmentedRepeatDataset(Dataset):
    """
    Virtually "expands" a dataset N-fold. Each access runs the transform,
    yielding a different random crop / photometric perturbation. This is
    equivalent to N differently-augmented copies.

    Compared to ConcatDataset([ds]*N):
      - Cleaner semantics: len = len(base) * num_views
      - No need to store N references in a list
      - DataLoader shuffle uniformly shuffles all N x len(base) virtual indices,
        without needing to worry about ConcatDataset internal segment boundaries

    Notes:
      - The underlying number of images is unchanged. num_views only changes
        "how many times each image is sampled per epoch".
      - Do not set num_views excessively high (e.g. >30) -- training on 600
        images 30 times per epoch risks overfitting. Recommend 5-15,
        monitored with val_loss.
    """

    def __init__(self, base_dataset, num_views=1):
        assert num_views >= 1, "num_views must be >= 1"
        self.base = base_dataset
        self.num_views = int(num_views)

    def __len__(self):
        return len(self.base) * self.num_views

    def __getitem__(self, idx):
        # Same base_idx is accessed num_views times, each with different random transform
        base_idx = idx % len(self.base)
        return self.base[base_idx]

    # Let outer code (e.g. Trainer logging) query the underlying info
    @property
    def dataset_type(self):
        return getattr(self.base, 'dataset_type', 'unknown')


# =============================================================================
# 5. Path_set_multi -- unchanged from previous version
# =============================================================================
def Path_set_multi(train_configs, test_configs,
                   train_split='train', test_split='test',
                   require_flow=True):
    def _collect(configs, split):
        img_list, flow_list, mask_list, pose_list, type_list = [], [], [], [], []
        for cfg in configs:
            ds_type = cfg['type']
            if ds_type not in DATASET_INTRINSICS:
                raise KeyError(f"Unknown dataset type '{ds_type}' in {split} config")

            scenes = sorted(glob(cfg['path']))
            if not scenes:
                print(f"[WARN] No scenes found for {ds_type} at {cfg['path']}")
                continue

            print(f"[Path_set_multi] {split} / {ds_type}: {len(scenes)} scenes "
                  f"from {cfg['path']}")

            for scene in scenes:
                scene_img_dir  = sorted(glob(f"{scene}/{split}_img/*"))
                scene_flow_dir = sorted(glob(f"{scene}/{split}_flow_sea/*"))
                scene_mask_dir = sorted(glob(f"{scene}/{split}_mask/*"))
                scene_pose_dir = sorted(glob(f"{scene}/{split}_pose/*"))

                if len(scene_pose_dir) == 0:
                    print(f"[WARN] {ds_type} {scene}: no pose under "
                          f"{split}_pose/* -- check split spelling?")
                    continue
                if len(scene_img_dir) != len(scene_pose_dir):
                    print(f"[WARN] {ds_type} {scene}: "
                          f"#img={len(scene_img_dir)} != #pose={len(scene_pose_dir)} -- skipping")
                    continue
                if require_flow and len(scene_flow_dir) != len(scene_pose_dir):
                    print(f"[WARN] {ds_type} {scene}: "
                          f"#flow={len(scene_flow_dir)} != #pose={len(scene_pose_dir)}")

                img_list  += scene_img_dir
                flow_list += scene_flow_dir
                mask_list += scene_mask_dir
                pose_list += scene_pose_dir
                type_list += [ds_type] * len(scene_pose_dir)

        assert len(img_list) == len(pose_list) == len(type_list), \
            (f"[{split}] list length mismatch: "
             f"imgs={len(img_list)}, poses={len(pose_list)}, types={len(type_list)}")
        return img_list, flow_list, mask_list, pose_list, type_list

    tr_img, tr_flow, tr_mask, tr_pose, tr_types = _collect(train_configs, train_split)
    te_img, te_flow, te_mask, te_pose, te_types = _collect(test_configs,  test_split)

    train_repeats = {}
    for cfg in train_configs:
        train_repeats[cfg['type']] = int(cfg.get('repeat', 1))
        if train_repeats[cfg['type']] < 1:
            raise ValueError(
                f"repeat must be >= 1 for dataset '{cfg['type']}', "
                f"got {train_repeats[cfg['type']]}"
            )

    for cfg in test_configs:
        if cfg.get('repeat', 1) != 1:
            print(f"[Path_set_multi] Ignoring repeat setting for test dataset "
                  f"'{cfg['type']}': validation/test does not use oversampling.")

    print("\n[Path_set_multi] summary:")
    for split_name, types, rp in [
        ('train', tr_types, train_repeats),
        ('test',  te_types, None),
    ]:
        c = Counter(types)
        if rp is not None:
            eff_count = {t: n * rp.get(t, 1) for t, n in c.items()}
            msg = ", ".join(
                f"{t}={n}(x{rp.get(t, 1)}->{eff_count[t]})" for t, n in c.items()
            )
            total_eff = sum(eff_count.values())
            print(f"  {split_name}: raw={len(types)}  effective={total_eff}  |  {msg}")
        else:
            msg = ", ".join(f"{t}={n}" for t, n in c.items())
            print(f"  {split_name}: total={len(types)}  |  {msg}")
    print()

    return {
        'train_img':     tr_img,    'train_flow':  tr_flow,
        'train_mask':    tr_mask,   'train_pose':  tr_pose,
        'train_types':   tr_types,
        'train_repeats': train_repeats,
        'test_img':      te_img,    'test_flow':   te_flow,
        'test_mask':     te_mask,   'test_pose':   te_pose,
        'test_types':    te_types,
    }


# =============================================================================
# 6. Transform factory -- supports aug_cfg configuration
# =============================================================================
DEFAULT_AUG_CFG = {
    'use_v2_crop':   True,    # True -> RandomResizeCropV2 (correct principal point)
                              # False -> old RandomResizeCrop (backward compat)
    'keep_center':   False,   # New default: spatial random crop ON
    'fix_ratio':     False,
    'max_scale':     2.5,
    'color_jitter': {         # None or {} means disabled
        'brightness': 0.0,
        'contrast':   0.0,
        'saturation': 0.0,
    },
}


def _merge_aug_cfg(user_cfg):
    """Merge user-provided aug_cfg with defaults. User settings take priority."""
    merged = {k: v for k, v in DEFAULT_AUG_CFG.items()}
    if user_cfg is None:
        return merged
    for k, v in user_cfg.items():
        if k == 'color_jitter' and isinstance(v, dict):
            merged['color_jitter'] = {**DEFAULT_AUG_CFG['color_jitter'], **v}
        else:
            merged[k] = v
    return merged


def build_train_transform(dataset_type, target_size=(480, 640), aug_cfg=None):
    """
    Build a training transform for a given dataset.

    Args:
        dataset_type:  'tartanair' / 'tartanair_shibuya' / ...
        target_size:   final (H, W)
        aug_cfg:       dict, see DEFAULT_AUG_CFG
    """
    aug = _merge_aug_cfg(aug_cfg)
    intrinsic = get_intrinsic_for(dataset_type)

    crop_cls = RandomResizeCropV2 if aug['use_v2_crop'] else RandomResizeCrop

    # If user wants keep_center=False but hasn't upgraded to V2, raise an error
    if (not aug['use_v2_crop']) and (not aug['keep_center']):
        raise ValueError(
            "[build_train_transform] keep_center=False requires use_v2_crop=True, "
            "otherwise the intrinsic channel principal point will be wrong. "
            "In your aug_cfg, set use_v2_crop: true or keep_center: true."
        )

    ops = [
        crop_cls(target_size, intrinsic,
                 max_scale=aug['max_scale'],
                 keep_center=aug['keep_center'],
                 fix_ratio=aug['fix_ratio'],
                 use_gpu=False),
    ]

    cj = aug.get('color_jitter') or {}
    if any(cj.get(k, 0) > 0 for k in ['brightness', 'contrast', 'saturation']):
        ops.append(ColorJitter(
            brightness=cj.get('brightness', 0.0),
            contrast=cj.get('contrast', 0.0),
            saturation=cj.get('saturation', 0.0),
        ))

    return Compose(ops)


def build_test_transform(dataset_type, target_size=(480, 640)):
    """Test transform: no augmentation, only CropCenter."""
    intrinsic = get_intrinsic_for(dataset_type)
    return Compose([
        CropCenter(target_size, intrinsic, use_gpu=False)
    ])


# =============================================================================
# 7. Multi-dataset training set assembly -- AugmentedRepeatDataset version
# =============================================================================
def build_multi_train_dataset(paths_dict, step=1, target_size=(480, 640),
                              max_scale=2.5, aug_cfg=None, logger=None):
    """
    Assemble a training ConcatDataset.

    Args:
        paths_dict:  dict returned by Path_set_multi
        step:        sampling stride
        target_size: final (H, W)
        max_scale:   used if aug_cfg does not specify max_scale (backward compat)
        aug_cfg:     augmentation config, see DEFAULT_AUG_CFG. None uses defaults.
        logger:      optional logger
    """
    # Backward compat: max_scale parameter can be overridden via aug_cfg
    aug_cfg = dict(aug_cfg) if aug_cfg else {}
    aug_cfg.setdefault('max_scale', max_scale)

    grouped = _group_by_type(
        paths_dict['train_img'],
        paths_dict['train_flow'],
        paths_dict['train_mask'],
        paths_dict['train_pose'],
        paths_dict['train_types'],
    )
    repeats = paths_dict.get('train_repeats', {})

    sub_datasets = []
    for ds_type, (imgs, flows, _masks, poses) in grouped.items():
        imgs  = imgs[::step]
        flows = flows[::step]
        poses = poses[::step]

        transform = build_train_transform(
            ds_type, target_size=target_size, aug_cfg=aug_cfg,
        )
        vo = VODataset(imgs, flows, None, poses, transform=transform)
        vo.dataset_type = ds_type

        repeat = repeats.get(ds_type, 1)
        if repeat > 1:
            # Wrap with AugmentedRepeatDataset instead of duplicating in ConcatDataset
            sub_datasets.append(AugmentedRepeatDataset(vo, num_views=repeat))
        else:
            sub_datasets.append(vo)

        if logger is not None:
            fx, fy, cx, cy = get_intrinsic_for(ds_type)
            cj = aug_cfg.get('color_jitter') or {}
            cj_on = any(cj.get(k, 0) > 0 for k in ['brightness', 'contrast', 'saturation'])
            aug_tags = []
            if not aug_cfg.get('keep_center', True):
                aug_tags.append('spatial_rand')
            if aug_cfg.get('max_scale', 2.5) > 1.0:
                aug_tags.append(f"scale<={aug_cfg['max_scale']}")
            if cj_on:
                aug_tags.append('color_jitter')
            aug_str = ','.join(aug_tags) if aug_tags else 'none'

            if repeat == 1:
                logger.info(
                    f"[MultiDataset] train / {ds_type}: {len(vo)} samples  "
                    f"(fx={fx:.1f}, cx={cx:.1f}, aug=[{aug_str}])"
                )
            else:
                logger.info(
                    f"[MultiDataset] train / {ds_type}: {len(vo)} x views={repeat} "
                    f"= {len(vo)*repeat} effective  "
                    f"(fx={fx:.1f}, cx={cx:.1f}, aug=[{aug_str}])"
                )

    if not sub_datasets:
        raise RuntimeError("No training sub-datasets built. Check your paths!")

    concat = ConcatDataset(sub_datasets)
    if logger is not None:
        unique_types = list(grouped.keys())
        logger.info(f"[MultiDataset] train total: {len(concat)} samples "
                    f"across {len(unique_types)} sub-datasets "
                    f"({', '.join(unique_types)})")
    return concat


# =============================================================================
# 8. Helper: group by dataset_type
# =============================================================================
def _group_by_type(imgs, flows, masks, poses, types):
    groups = defaultdict(lambda: ([], [], [], []))
    for i, t in enumerate(types):
        groups[t][0].append(imgs[i])
        groups[t][1].append(flows[i] if i < len(flows) else None)
        groups[t][2].append(masks[i] if i < len(masks) else None)
        groups[t][3].append(poses[i])
    return dict(groups)
