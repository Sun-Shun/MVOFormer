import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from Tool.Utils.utils import tensor_reshape_gpu


class DINOv3(nn.Module):
    """DINOv3 pre-trained model wrapper for visual feature extraction."""

    def __init__(self, version='small', freeze=True,
                 weights_dir='Network/dinov3/weights/', device='cuda'):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        BACKBONE_SIZE = version
        backbone_archs = {
            "small":     "vits16",
            "smallplus": "vits16plus",
            "base":      "vitb16",
        }
        weights_list = {
            "small":     "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "smallplus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            "base":      "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        }

        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov3_{backbone_arch}"
        weight = weights_dir + weights_list[BACKBONE_SIZE]

        self.dinov3 = torch.hub.load('./Network/dinov3', backbone_name,
                                     source='local', weights=weight).to(self.device)

        self.freeze = freeze
        self.pre_DINOv3_features = None

    def reset_rnn_state(self):
        """Clear cached previous-frame features so the next forward processes both images."""
        self.pre_DINOv3_features = None

    def warp_feature(self, x, flo):
        """
        Warp feature map x according to optical flow flo.

        The flow is downsampled to match the feature resolution and rescaled,
        avoiding the memory explosion of upsampling features to image resolution.

        Args:
            x:   [B, C, H_x, W_x] feature map
            flo: [B, 2, H_f, W_f] optical flow in pixel units at image resolution
        Returns:
            output:     [B, C, H_x, W_x] warped feature
            final_mask: [B, 1, H_x, W_x] binary validity mask
        """
        B, C, H, W = x.size()
        _, _, H_f, W_f = flo.size()

        # Downsample flow to feature resolution with proportional displacement scaling
        if (H_f, W_f) != (H, W):
            scale_w = float(W) / float(W_f)
            scale_h = float(H) / float(H_f)
            flo_small = F.interpolate(flo, size=(H, W), mode='bilinear', align_corners=True)
            flo_small = torch.stack([
                flo_small[:, 0] * scale_w,
                flo_small[:, 1] * scale_h,
            ], dim=1)
        else:
            flo_small = flo

        # Build sampling grid at feature resolution
        xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        grid = torch.cat((xx, yy), 1)

        vgrid = grid - flo_small.float()
        original_vgrid = vgrid

        # Normalize to [-1, 1] for grid_sample
        vgrid_norm_x = 2.0 * vgrid[:, 0:1] / max(W - 1, 1) - 1.0
        vgrid_norm_y = 2.0 * vgrid[:, 1:2] / max(H - 1, 1) - 1.0
        vgrid_norm = torch.cat([vgrid_norm_x, vgrid_norm_y], dim=1)
        vgrid_norm = vgrid_norm.permute(0, 2, 3, 1)

        output = F.grid_sample(x, vgrid_norm.to(x.dtype), align_corners=True)

        # Validity mask: pixels sampled from outside the feature boundaries
        ones_mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)
        mask = F.grid_sample(ones_mask, vgrid_norm.to(x.dtype),
                             mode='bilinear', padding_mode='zeros', align_corners=True)

        valid_x = (original_vgrid[:, 0:1] >= 0) & (original_vgrid[:, 0:1] < W)
        valid_y = (original_vgrid[:, 1:2] >= 0) & (original_vgrid[:, 1:2] < H)
        coordinate_mask = (valid_x & valid_y).to(x.dtype)

        final_mask = mask * coordinate_mask
        final_mask = (final_mask > 0.5).to(x.dtype)
        output = output * final_mask
        return output, final_mask

    def infer_img(self, images):
        if self.freeze:
            with torch.no_grad():
                features = self.dinov3.get_intermediate_layers(images, n=4, reshape=True)
        else:
            features = self.dinov3.get_intermediate_layers(images, n=4, reshape=True)
        return features

    @staticmethod
    def make_transform(resize_size: int = 256):
        to_tensor = T.ToImage()
        resize = T.Resize((resize_size, resize_size), antialias=True)
        to_float = T.ToDtype(torch.float32, scale=True)
        normalize = T.Normalize(
            mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143),
        )
        return T.Compose([to_tensor, resize, to_float, normalize])

    def forward(self, inputs, flow=None, input_size=518, rnn=None):
        if self.pre_DINOv3_features is None:
            inf_imgs = torch.cat((inputs[0], inputs[1]), dim=0)
            images = tensor_reshape_gpu(inf_imgs, input_size, self.device)
            bs, _, h, w = images.shape
            bs = bs // 2
            dino_features = self.infer_img(images)
            dino_feature = dino_features[-1]
            self.pre_DINOv3_features = dino_feature[:bs, :, :]
            features = dino_feature[bs:, :, :]
        else:
            inf_imgs = inputs[1]
            images = tensor_reshape_gpu(inf_imgs, input_size, self.device)
            bs, _, h, w = images.shape
            dino_features = self.infer_img(images)
            dino_feature = dino_features[-1]
            features = dino_feature

        k_feature = features

        # Warp and cosine similarity at feature resolution
        if flow is not None:
            warp_DINOv3_features, final_mask = self.warp_feature(
                self.pre_DINOv3_features.to(self.device),
                flow.to(self.device),
            )
            cosine_feature = F.cosine_similarity(
                k_feature, warp_DINOv3_features, dim=1).detach()
        else:
            cosine_feature = None
            final_mask = None

        pre_features = self.pre_DINOv3_features

        if rnn:
            self.pre_DINOv3_features = k_feature
        else:
            self.pre_DINOv3_features = None

        return k_feature, pre_features, cosine_feature, final_mask
