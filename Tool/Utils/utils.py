import os
import io
import cv2
import torch
import logging
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional
import torch.nn.functional as F
import torchvision.transforms as T


def is_main_process():
    return True


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# ----------------------------------------------------------------------------
# DINOv3 image preprocessing helpers (merged from transform_img.py)
# ----------------------------------------------------------------------------
def resize_with_aspect_ratio(imgs, target_size, ensure_multiple_of=14,
                             resize_method='lower_bound', interpolation='bilinear'):
    """
    Resize a batch of images while maintaining aspect ratio, ensuring dimensions
    are multiples of ensure_multiple_of. Operates entirely on GPU.

    Args:
        imgs: Tensor of shape (B, C, H, W)
        target_size: Target size for resizing
        ensure_multiple_of: Ensure dimensions are multiples of this value
        resize_method: 'lower_bound' or other method for scaling
        interpolation: Interpolation method

    Returns:
        Resized images tensor of shape (B, C, new_H, new_W)
    """
    B, C, H, W = imgs.shape

    if resize_method == 'lower_bound':
        scales = torch.min(target_size / torch.tensor([H, W], device=imgs.device), dim=0)[0]
    else:
        scales = target_size / torch.max(torch.tensor([H, W], device=imgs.device), dim=0)[0]

    new_H = (H * scales).int()
    new_W = (W * scales).int()

    if ensure_multiple_of:
        new_H = (new_H // ensure_multiple_of) * ensure_multiple_of
        new_W = (new_W // ensure_multiple_of) * ensure_multiple_of

    imgs_resized = F.interpolate(imgs, size=(new_H, new_W), mode=interpolation, align_corners=False)
    return imgs_resized


def tensor_reshape_gpu(images, input_size=518, DEVICE='cuda'):
    """
    Process a batch of images on GPU with resizing and normalization.

    Args:
        images: Tensor of shape (B, C, H, W) in BGR format
        input_size: Target size for resizing
        DEVICE: Device to perform computations on

    Returns:
        Processed images tensor of shape (B, C, new_H, new_W)
    """
    transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = images.to(DEVICE)
    images = images / 255.0
    images_resized = resize_with_aspect_ratio(images, input_size, ensure_multiple_of=16)
    processed_images = transform(images_resized)
    return processed_images


def create_logger(log_file, rank=0):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO if rank == 0 else 'ERROR',
                        format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else 'ERROR')
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def apply_dataset_ratio(paths, ratio, seed=666):
    """
    Randomly sample a fraction of training data while keeping lists consistent.

    Args:
        paths: list of lists [img, flow, mask, pose]
        ratio: float in (0, 1], fraction of data to use
        seed:  random seed for reproducibility
    Returns:
        Truncated lists with the same format.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"train_ratio must be in (0, 1], got {ratio}")
    if ratio == 1.0:
        return paths

    n_total = next((len(lst) for lst in paths if lst), 0)
    if n_total == 0:
        return paths

    n_keep = max(1, int(n_total * ratio))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n_total), n_keep))

    return [[lst[i] for i in indices] if lst else lst for lst in paths]


def my_worker_init_fn(worker_id):
    """Prevent OpenCV from spawning threads in DataLoader workers (avoids deadlocks)."""
    import cv2
    import numpy as np
    import random
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(seed)

def visualize_uncertainty(input_image, uncertainty_map, alpha=0.6, save_path="uncertainty.png",
                           quantile_upper=1, deepen_factor=2.0, cmap="hot"):
    """Resize and overlay uncertainty map onto the original image.

    Args:
        input_image:     tensor (B, 3, H, W)
        uncertainty_map: tensor (B, 1, h, w) or (B, h, w) or (h, w)
        alpha:           overlay transparency
        quantile_upper:  upper quantile for clipping extreme values
        deepen_factor:   exponent to emphasize high-uncertainty regions
        cmap:            colormap name
    """
    H, W = input_image.shape[2], input_image.shape[3]

    unc = uncertainty_map.detach().cpu()
    if unc.dim() == 4:
        unc = unc[0, 0]
    elif unc.dim() == 3:
        unc = unc[0]
    unc_np = unc.numpy().astype(np.float32)

    unc_resized = cv2.resize(unc_np, (W, H), interpolation=cv2.INTER_LINEAR)

    upper = np.quantile(unc_resized, quantile_upper)
    unc_clipped = np.clip(unc_resized, a_min=0.0, a_max=upper)
    vmin, vmax = unc_clipped.min(), unc_clipped.max()
    if vmax > vmin:
        unc_normalized = (unc_clipped - vmin) / (vmax - vmin)
    else:
        unc_normalized = np.zeros_like(unc_clipped)
    unc_deepened = np.power(unc_normalized, deepen_factor)

    plt.figure(figsize=(W / 100, H / 100), dpi=100)
    sns.heatmap(unc_deepened, cmap=cmap, cbar=False)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    heatmap_image = np.array(Image.open(buf).convert("RGB"))
    plt.close()

    heatmap_resized = cv2.resize(heatmap_image, (W, H), interpolation=cv2.INTER_LINEAR)

    image = input_image[0].detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    original_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_PLASMA)
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)

def visualize_mask_weight(input_image, mask_weight, feat_size=None,
                          save_path="mask_weight.png", alpha=0.5, cmap="viridis",
                          dynamic_threshold=0.3, title_prefix="", vmin=None, vmax=None):
    """
    4-panel diagnostic plot for mask weight visualization.

    Panels:
        [1] Original image
        [2] Mask heatmap + colorbar
        [3] Mask overlay on image
        [4] Histogram with mean/median/threshold reference lines
    """
    m = mask_weight.detach().cpu().float()
    if m.dim() == 1:
        m = m.unsqueeze(0)
    if m.dim() == 4:
        m = m[0, 0]
    elif m.dim() == 3:
        m = m[0]
    elif m.dim() == 2:
        L = m.shape[1]
        H_img, W_img = input_image.shape[2], input_image.shape[3]
        if feat_size is None:
            for stride in [14, 16, 8, 32]:
                if (H_img // stride) * (W_img // stride) == L:
                    feat_size = (H_img // stride, W_img // stride)
                    break
            if feat_size is None:
                sq = int(round(math.sqrt(L)))
                feat_size = (sq, max(1, L // sq))
        m = m[0].reshape(feat_size)
    m_np = m.numpy().astype(np.float32)

    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = max(1.0, float(m_np.max()))

    img = input_image[0].detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    H_img, W_img = img.shape[:2]

    mask_resized = cv2.resize(m_np, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

    stats = {
        "min": float(m_np.min()),
        "max": float(m_np.max()),
        "mean": float(m_np.mean()),
        "median": float(np.median(m_np)),
        "std": float(m_np.std()),
        "p10": float(np.quantile(m_np, 0.10)),
        "p90": float(np.quantile(m_np, 0.90)),
        "frac_below_thr": float((m_np < dynamic_threshold).mean()),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"{title_prefix}img1 (original)", fontsize=11)
    axes[0, 0].axis('off')

    im = axes[0, 1].imshow(m_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 1].set_title(
        f"mask heatmap  (feat {m_np.shape[0]}x{m_np.shape[1]})\n"
        f"low -> dynamic (suppressed)   high -> static", fontsize=10)
    axes[0, 1].axis('off')
    cbar = plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar.set_label("mask value", fontsize=9)

    norm_for_overlay = np.clip((mask_resized - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    cmap_fn = plt.get_cmap(cmap)
    mask_colored = cmap_fn(norm_for_overlay)[:, :, :3]
    mask_colored = (mask_colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 1.0 - alpha, mask_colored, alpha, 0)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(
        f"mask overlay on img1  (alpha={alpha})\n"
        f"dark / cold regions = strongly suppressed", fontsize=10)
    axes[1, 0].axis('off')

    ax_h = axes[1, 1]
    ax_h.hist(m_np.ravel(), bins=60, range=(vmin, vmax),
              color='steelblue', alpha=0.75, edgecolor='black', linewidth=0.5)
    ax_h.axvline(stats["mean"], color='red', linestyle='--', linewidth=1.5,
                 label=f"mean={stats['mean']:.3f}")
    ax_h.axvline(stats["median"], color='green', linestyle='--', linewidth=1.5,
                 label=f"median={stats['median']:.3f}")
    ax_h.axvline(dynamic_threshold, color='black', linestyle=':', linewidth=1.5,
                 label=f"thr={dynamic_threshold}")
    ax_h.set_xlabel("mask value")
    ax_h.set_ylabel("count")
    ax_h.set_title(
        f"distribution  --  frac<{dynamic_threshold}: {stats['frac_below_thr']*100:.1f}%,  "
        f"std={stats['std']:.3f}", fontsize=10)
    ax_h.legend(loc='best', fontsize=9)
    ax_h.grid(True, alpha=0.3)
    ax_h.set_xlim(vmin, vmax)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return stats


def uncertainty_to_mask_mult(uncertainty, clamp_max=3.0, unc_min=0.1):
    """
    Reproduce the mask_mult computation from UncertaintyModule:
        loss_mult = 1 / (2 * u^2)
        mask_mult = loss_mult.clamp_max(3)

    Args:
        uncertainty: tensor of any shape
        clamp_max:   upper clamp value for mask
        unc_min:     lower clamp for uncertainty (prevent division by zero)
    Returns:
        mask_mult: same shape as uncertainty
    """
    u = uncertainty.clamp(min=unc_min)
    mask = 1.0 / (2.0 * u.pow(2))
    mask = mask.clamp_max(clamp_max)
    return mask


def visualize_mask_from_uncertainty(
        input_image, uncertainty, save_path="mask_from_uncertainty.png",
        clamp_max=3.0, unc_min=0.1, alpha=0.5, cmap="viridis",
        dynamic_threshold_ratio=0.66, title_prefix=""):
    """
    Compute mask_mult from uncertainty using the model formula, then draw 4-panel plot.

    This is the recommended usage -- uses outputs['cosine_uncertainty'] directly
    without modifying the model forward.

    Args:
        input_image:             (B, 3, H, W) tensor
        uncertainty:             (B, 1, h, w) / (B, h, w) / (h, w)
        save_path:               output 4-panel image path
        clamp_max, unc_min:      consistent with UncertaintyModule params
        dynamic_threshold_ratio: dynamic threshold = clamp_max * ratio (default 0.66)
    Returns:
        stats: dict
    """
    mask = uncertainty_to_mask_mult(uncertainty, clamp_max=clamp_max, unc_min=unc_min)
    dynamic_threshold = clamp_max * dynamic_threshold_ratio

    return visualize_mask_weight(
        input_image=input_image, mask_weight=mask, feat_size=None,
        save_path=save_path, alpha=alpha, cmap=cmap,
        dynamic_threshold=dynamic_threshold,
        title_prefix=title_prefix + f"[mask = clamp(1/(2*u^2), {clamp_max})]  ",
        vmin=0.0, vmax=clamp_max)
