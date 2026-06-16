from __future__ import division
import torch
import math
import random
import numpy as np
import numbers
import matplotlib.pyplot as plt
import torch.nn.functional as F




# ----------------------------------------------------------------------------
# Internal utility functions
# ----------------------------------------------------------------------------
def _is_in_dataloader_worker():
    """
    [Fix 6] Detect whether we are running inside a DataLoader worker process.
    Worker child processes cannot safely access CUDA, so transforms must
    auto-downgrade to CPU.
    """
    try:
        info = torch.utils.data.get_worker_info()
        return info is not None
    except Exception:
        return False


def _ensure_float(t: torch.Tensor) -> torch.Tensor:
    """
    [Fix 5] Promote uint8 image tensor to float32 [0, 255].
    Flow/mask etc. are already float32 and are left unchanged.
    F.interpolate does not accept uint8.

    Input:  (C, H, W), arbitrary dtype
    Output: (C, H, W), float32 (if originally uint8 -> float32 but keeps [0,255]
            range, since ToTensor handles /255.0 normalization later)
    """
    if t.dtype == torch.uint8:
        return t.float()
    if t.dtype != torch.float32:
        return t.float()
    return t


# ============================================================================
# General transforms
# ============================================================================
class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def generate_random_scale_crop(h, w, scale_base, keep_center, fix_ratio):
    scale_s_w = random.random() * (scale_base - 1) + 1
    if fix_ratio:
        scale_s_h = scale_s_w
    else:
        scale_s_h = random.random() * (scale_base - 1) + 1

    n_w = scale_s_w * w
    n_h = scale_s_h * h

    scale_c_w = 0.5 * random.random() + 1.0
    scale_c_h = 0.5 * random.random() + 1.0
    crop_w = int(math.ceil(n_w / scale_c_w))
    crop_h = int(math.ceil(n_h / scale_c_h))

    if keep_center:
        x1 = int((n_w - crop_w) / 2)
        y1 = int((n_h - crop_h) / 2)
    else:
        x1 = random.randint(0, int(n_w - crop_w))
        y1 = random.randint(0, int(n_h - crop_h))

    return scale_s_w, scale_s_h, x1, y1, crop_w, crop_h


# ============================================================================
# RandomResizeCrop  /  CropCenter  (with intrinsic tracking)
# ============================================================================
class RandomResizeCrop(object):
    """
    Random scale + crop with optional GPU acceleration.

    [Fix 6] use_gpu=True auto-downgrades to False inside DataLoader workers.
    So during normal training (num_workers>0), this transform runs entirely on
    CPU, which is expected: GPU acceleration is done by Trainer._to_device in
    the main process.
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
        # [Fix 5] uint8 -> float (F.interpolate doesn't accept uint8)
        for kk in sample:
            if isinstance(sample[kk], torch.Tensor) and sample[kk].dim() == 3:
                sample[kk] = _ensure_float(sample[kk])

        # [Fix 6] Worker processes cannot use CUDA, auto-downgrade
        in_worker = _is_in_dataloader_worker()
        use_gpu_eff = self.use_gpu and (not in_worker) and torch.cuda.is_available()
        device = torch.device('cuda') if use_gpu_eff else torch.device('cpu')

        if use_gpu_eff:
            for kk in sample:
                if isinstance(sample[kk], torch.Tensor):
                    sample[kk] = sample[kk].to(device)

        for kk in sample:
            if isinstance(sample[kk], torch.Tensor) and sample[kk].dim() >= 2:
                h, w = sample[kk].shape[1], sample[kk].shape[2]
                break

        scale_w, scale_h, x1, y1, crop_w, crop_h = generate_random_scale_crop(
            h, w, self.scale_base, self.keep_center, self.fix_ratio
        )

        scale_h_2, scale_w_2, scale_2 = 1., 1., 1.
        x1_2 = int((crop_w - self.target_w) / 2)
        y1_2 = int((crop_h - self.target_h) / 2)
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

        for nn in sample:
            if nn not in ['pose']:
                sample[nn] = F.interpolate(
                    sample[nn].unsqueeze(0), scale_factor=(scale_h, scale_w),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                sample[nn] = sample[nn][:, y1:y1 + crop_h, x1:x1 + crop_w]
                if scale_2 > 1.0:
                    sample[nn] = F.interpolate(
                        sample[nn].unsqueeze(0), size=(crop_h_2, crop_w_2),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)
                sample[nn] = sample[nn][:, y1_2:y1_2 + int(self.target_h), x1_2:x1_2 + int(self.target_w)]

        sample['intrinsic'] = make_intrinsics_layer(
            self.target_w, self.target_h,
            self.fx * scale_w * scale_2, self.fy * scale_h * scale_2,
            self.target_w / 2.0, self.target_h / 2.0,
            use_gpu=use_gpu_eff,
        )
        if use_gpu_eff:
            sample['intrinsic'] = sample['intrinsic'].to(device)

        if 'flow' in sample:
            sample['flow'][0, :, :] = sample['flow'][0, :, :] * scale_w * scale_2
            sample['flow'][1, :, :] = sample['flow'][1, :, :] * scale_h * scale_2

        # [Fix 1] Return CPU tensors after compute (pin_memory compat)
        if use_gpu_eff:
            for kk in sample:
                if isinstance(sample[kk], torch.Tensor):
                    sample[kk] = sample[kk].cpu()

        return sample


class CropCenter(object):
    """
    Center crop; resizes first if image is too small.

    [Fix 4] Correctly tracks principal point offset after scale + center-crop.
    [Fix 6] use_gpu auto-downgrades to CPU inside workers.
    """

    # [Fix 2] Add missing use_gpu parameter
    def __init__(self, size, intrinsic, use_gpu=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fx = intrinsic[0]
        self.fy = intrinsic[1]
        self.cx = intrinsic[2]
        self.cy = intrinsic[3]
        self.use_gpu = use_gpu
        self._intrinsic_logged = False

    def __call__(self, sample):
        # [Fix 5] uint8 -> float
        for kk in sample:
            if isinstance(sample[kk], torch.Tensor) and sample[kk].dim() == 3:
                sample[kk] = _ensure_float(sample[kk])

        # [Fix 6] Worker processes cannot use CUDA
        in_worker = _is_in_dataloader_worker()
        use_gpu_eff = self.use_gpu and (not in_worker) and torch.cuda.is_available()
        device = torch.device('cuda') if use_gpu_eff else torch.device('cpu')

        if use_gpu_eff:
            for kk in sample:
                if isinstance(sample[kk], torch.Tensor):
                    sample[kk] = sample[kk].to(device)

        kks = list(sample.keys())
        th, tw = self.size
        h, w = sample[kks[0]].shape[1], sample[kks[0]].shape[2]

        # One-time diagnostic log [Fix 4]
        if not self._intrinsic_logged:
            is_centered = (abs(self.cx - w / 2.0) < 1.0 and
                           abs(self.cy - h / 2.0) < 1.0)
            if is_centered:
                print(
                    f"[CropCenter] Original {w}x{h}, principal point (cx={self.cx:.2f}, "
                    f"cy={self.cy:.2f}) is at image center -- intrinsic tracking "
                    f"equivalent to old version."
                )
            else:
                print(
                    f"[CropCenter] Original {w}x{h}, principal point (cx={self.cx:.2f}, "
                    f"cy={self.cy:.2f}) is NOT at center (center should be "
                    f"{w/2:.1f}, {h/2:.1f}). [Fix 4] Tracking offset correctly."
                )
            self._intrinsic_logged = True

        # Step 1: if target > original, upscale
        scale_h, scale_w, scale = 1., 1., 1.
        if th > h:
            scale_h = float(th) / h
        if tw > w:
            scale_w = float(tw) / w
        if scale_h > 1 or scale_w > 1:
            scale = max(scale_h, scale_w)
            w = int(round(w * scale))
            h = int(round(h * scale))

        # Step 2: center crop
        x1 = int((w - tw) / 2)
        y1 = int((h - th) / 2)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape) == 3:
                if scale > 1:
                    img = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
                    img = img.squeeze(0)
                sample[kk] = img[:, y1:y1 + th, x1:x1 + tw]

        # [Fix 4] Correct principal point tracking
        fx_final = self.fx * scale
        fy_final = self.fy * scale
        cx_final = self.cx * scale - x1
        cy_final = self.cy * scale - y1

        sample['intrinsic'] = make_intrinsics_layer(
            tw, th, fx_final, fy_final, cx_final, cy_final,
            use_gpu=use_gpu_eff,
        )
        if use_gpu_eff:
            sample['intrinsic'] = sample['intrinsic'].to(device)

        if "flow" in sample:
            sample["flow"][0, :, :] = sample["flow"][0, :, :] * scale
            sample["flow"][1, :, :] = sample["flow"][1, :, :] * scale

        # [Fix 3] Return CPU tensors
        if use_gpu_eff:
            for kk in sample:
                if isinstance(sample[kk], torch.Tensor):
                    sample[kk] = sample[kk].cpu()

        return sample


class DownscaleFlow(object):
    """Scale the flow and mask to a fixed size."""

    def __init__(self, scale=4):
        self.downscale = 1.0 / scale

    def __call__(self, sample):
        if self.downscale != 1 and 'flow' in sample:
            sample['flow'] = F.interpolate(
                sample['flow'].unsqueeze(0), scale_factor=self.downscale,
                mode='bilinear', align_corners=False
            ).squeeze(0)

        if self.downscale != 1 and 'intrinsic' in sample:
            sample['intrinsic'] = F.interpolate(
                sample['intrinsic'].unsqueeze(0), scale_factor=self.downscale,
                mode='bilinear', align_corners=False
            ).squeeze(0)

        return sample


class ToTensor(object):
    """
    [Fix 7] Compatible with torch.Tensor input (new VODataest.py returns tensors).

    Behaviour:
      - If data is torch.Tensor:
          shape (H, W)   -> add channel dim (1, H, W); preserve dtype
          shape (3, H, W) -> divide by 255.0, auto-promote to float32 (uint8-safe)
          other -> no change
      - If data is np.ndarray (backward compat):
          shape (H, W)   -> add channel dim
          shape (3, H, W) -> divide by 255.0
          other -> no change
    """

    def __call__(self, sample):
        kks = list(sample)
        for kk in kks:
            data = sample[kk]
            if data is None:
                continue

            if isinstance(data, torch.Tensor):
                if data.dim() == 2:
                    data = data.unsqueeze(0)
                elif data.dim() == 3 and data.shape[0] == 3:
                    # uint8-safe: /255.0 auto-promotes to float
                    data = data.float() / 255.0
                # other cases (flow/mask/intrinsic): no change
                sample[kk] = data
            elif isinstance(data, np.ndarray):
                if data.ndim == 2:
                    data = data.reshape((1,) + data.shape)
                elif data.ndim == 3 and data.shape[0] == 3:
                    data = data / 255.0
                sample[kk] = data
            # other types (e.g. int/float scalars): no change

        return sample


# ============================================================================
# Misc utilities
# ============================================================================




def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4, 4))
    plt.subplot(111)
    if gtposes is not None:
        plt.plot(gtposes[:, 0], gtposes[:, 1], linestyle='dashed', c='k')
    plt.plot(estposes[:, 0], estposes[:, 1], c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    legend_labels = []
    if gtposes is not None:
        legend_labels.append('Ground Truth')
    legend_labels.append('MVOFormer')
    plt.legend(legend_labels)
    plt.title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)


def make_intrinsics_layer(w, h, fx, fy, ox, oy, use_gpu=False):
    """
    Build a per-pixel intrinsic encoding layer.

    [Fix 6] use_gpu defaults to False. Worker processes cannot use CUDA.
    Only set use_gpu=True when the caller is known to be the main process.
    """
    if use_gpu and torch.cuda.is_available():
        # Only for main process callers that explicitly know this is safe
        hh, ww = torch.meshgrid(
            torch.arange(h, device='cuda'),
            torch.arange(w, device='cuda'),
            indexing='ij',
        )
    else:
        hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    ww = (ww.float() - ox + 0.5) / fx
    hh = (hh.float() - oy + 0.5) / fy
    intrinsicLayer = torch.stack((ww, hh), dim=-1).permute(2, 0, 1)
    return intrinsicLayer


