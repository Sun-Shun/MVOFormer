import torch
import copy
import torch.nn.functional as F

from torch import nn
from Tool.Utils.utils import NestedTensor
from Network.Model.backbone import build_backbone
from Network.Model.fp_transformer import build_fp_transformer
from Network.Model.dinov3_module import DINOv3


class MLP(nn.Module):
    """Simple multi-layer perceptron (FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes),
        nn.GELU())


def linear_no(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), )


class Conv_Head(nn.Module):
    def __init__(self, hidden_dim, num_queries, num_pred=3, aux_loss=False):
        super().__init__()

        self.conv1d_layer = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim // 2,
                                       kernel_size=3, stride=1, padding=1)
        fc1_trans = linear_no(hidden_dim // 2 * num_queries, 128)
        fc2_trans = linear(128, 32)
        fc3_trans = nn.Linear(32, 3)

        fc1_rot = linear_no(hidden_dim // 2 * num_queries, 128)
        fc2_rot = linear(128, 32)
        fc3_rot = nn.Linear(32, 3)

        # Zero-initialize the final linear layer so the network starts by
        # predicting identity pose (residual = 0), avoiding NaN in early training.
        nn.init.zeros_(fc3_trans.weight)
        nn.init.zeros_(fc3_trans.bias)
        nn.init.zeros_(fc3_rot.weight)
        nn.init.zeros_(fc3_rot.bias)

        self.MLP_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.MLP_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)

        if aux_loss:
            self.conv1d_layer = _get_clones(self.conv1d_layer, num_pred)
            self.MLP_trans = _get_clones(self.MLP_trans, num_pred)
            self.MLP_rot = _get_clones(self.MLP_rot, num_pred)
        else:
            self.conv1d_layer = nn.ModuleList([self.conv1d_layer for _ in range(num_pred)])
            self.MLP_trans = nn.ModuleList([self.MLP_trans for _ in range(num_pred)])
            self.MLP_rot = nn.ModuleList([self.MLP_rot for _ in range(num_pred)])

    def forward(self, x):
        """
        Args:
            x: (L, B, Q, C) list of decoder layer outputs.
               L = num_pred, B = batch, Q = queries, C = hidden_dim.
        Returns:
            trans_output, rot_output: lists of (B, 3) cumulative predictions.
        """
        trans_output = []
        rot_output = []

        bs = x[0].shape[0]
        device = x[0].device
        dtype = x[0].dtype
        trans = torch.zeros(bs, 3, device=device, dtype=dtype)
        rot = torch.zeros(bs, 3, device=device, dtype=dtype)

        for i in range(len(x)):
            x_conv = self.conv1d_layer[i](x[i].permute(0, 2, 1))
            x_conv = x_conv.flatten(start_dim=1)

            # Non-in-place addition ensures each appended tensor is independent
            trans = trans + self.MLP_trans[i](x_conv)
            rot = rot + self.MLP_rot[i](x_conv)

            trans_output.append(trans)
            rot_output.append(rot)

        return trans_output, rot_output


class VOTransformer(nn.Module):
    def __init__(self, backbone, fp_transformer, with_pose_refine, num_queries, num_feature_levels,
                 aux_loss=True, init_pose=False, is_Semantics=False, DINOv3_version='small',
                 DINOv3_weights_dir='Network/dinov3/weights/'):
        super().__init__()

        self.num_queries = num_queries

        if is_Semantics:
            self.dinov3 = DINOv3(version=DINOv3_version,
                                 weights_dir=DINOv3_weights_dir)
            self.dinov3_embed_dim = self.dinov3.dinov3.embed_dim
        else:
            self.dinov3 = None
            self.dinov3_embed_dim = None

        self.fp_transformer = fp_transformer
        self.fp_transformer.DINOv3_embed_dim = self.dinov3_embed_dim

        hidden_dim = fp_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.with_pose_refine = with_pose_refine

        self.delta_ref_point = MLP(hidden_dim, hidden_dim, 3, 3)
        if init_pose == True:
            nn.init.constant_(self.delta_ref_point.layers[-1].weight.data, 0)
            nn.init.constant_(self.delta_ref_point.layers[-1].bias.data, 0)

        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (fp_transformer.decoder.num_layers + 1)
        self.num_pred = num_pred

        self.fp_transformer.decoder.with_pose_refine = self.with_pose_refine
        self.delta_ref_point = _get_clones(self.delta_ref_point, num_pred)
        self.fp_transformer.decoder.delta_ref_point = self.delta_ref_point

        self.head = Conv_Head(hidden_dim=hidden_dim, num_queries=num_queries, num_pred=num_pred,
                              aux_loss=self.aux_loss)

    def forward(self, x_list, scale_factor=4, rnn_time=False):
        imgs = [x_list[0], x_list[1]]
        flow = x_list[2]
        intrinsic_flow = torch.cat((flow, x_list[3]), dim=1)
        intrinsic_flow = F.interpolate(intrinsic_flow, scale_factor=1.0 / scale_factor, mode='bilinear',
                                       align_corners=False)

        features, pos = self.backbone(intrinsic_flow)

        if self.dinov3 is not None:
            k_feature, pre_k_feature, cosine_feature, final_mask = self.dinov3(
                imgs, flow, input_size=518, rnn=rnn_time)
            DINOv3_feature = [k_feature, pre_k_feature]
            cosine_feature_mask = [cosine_feature, final_mask]
        else:
            DINOv3_feature = None
            cosine_feature_mask = None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.training:
            query_embeds = self.query_embed.weight
        else:
            query_embeds = self.query_embed.weight[:self.num_queries]

        hs, init_reference, inter_references, uncertainty = \
            self.fp_transformer(srcs, masks, pos, query_embeds, DINOv3_feature)

        if uncertainty is None:
            uncertainty = [None] * self.num_pred
        cosine_feature_mask_list = [cosine_feature_mask] * self.num_pred

        out = {}

        outputs_pose_translations, outputs_pose_rots = self.head(hs)
        outputs_pose_translations = torch.stack(outputs_pose_translations)
        outputs_pose_rots = torch.stack(outputs_pose_rots)

        out['outputs_pose_translations'] = outputs_pose_translations[-1]
        out['outputs_pose_rots'] = outputs_pose_rots[-1]
        out['cosine_uncertainty'] = uncertainty[-1]
        out['cosine_feature_mask'] = cosine_feature_mask_list[-1]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_pose_translations, outputs_pose_rots,
                uncertainty, cosine_feature_mask_list)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_pose_translations, outputs_pose_rots, uncertainty, cosine_feature_mask):
        return [
            {'outputs_pose_translations': a, 'outputs_pose_rots': b,
             'cosine_uncertainty': c, 'cosine_feature_mask': d}
            for a, b, c, d in zip(outputs_pose_translations[:-1], outputs_pose_rots[:-1],
                                  uncertainty[:-1], cosine_feature_mask[:-1])]


class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses, uncertainty_regularizer_weight=0.5):
        super().__init__()
        self.weight_dict = weight_dict
        self.uncertainty_regularizer_weight = uncertainty_regularizer_weight
        self.losses = losses
        self.criterion = nn.L1Loss()

    def loss_pose(self, outputs, targets):
        """
        x: [flow, intrinsic, pose]
        y: pose
        """
        gt_pose = targets
        pose_trans = outputs['outputs_pose_translations']
        pose_rot = outputs['outputs_pose_rots']

        # Safe normalization: clamp to avoid division by zero
        pose_trans_f32 = pose_trans.float()
        trans_norm_den = torch.norm(pose_trans_f32, dim=1, keepdim=True).clamp(min=1e-6)
        tran_norm = (pose_trans_f32 / trans_norm_den).to(pose_trans.dtype)

        loss_tran = self.criterion(tran_norm, gt_pose[:, :3])
        loss_rot = self.criterion(pose_rot, gt_pose[:, 3:])

        losses = {}
        losses['loss_trans'] = loss_tran
        losses['loss_rot'] = loss_rot
        return losses

    def loss_uncert(self, outputs, targets):
        uncertainty = outputs['cosine_uncertainty']
        cosine_feature_mask = outputs['cosine_feature_mask']

        if uncertainty is None or cosine_feature_mask is None:
            losses = {'loss_uncertainty': torch.tensor(0.0, device=targets.device, requires_grad=True)}
            return losses

        cosine_feature, cosine_mask = cosine_feature_mask
        if cosine_feature is None:
            losses = {'loss_uncertainty': torch.tensor(0.0, device=targets.device, requires_grad=True)}
            return losses

        uncertainty = uncertainty.clamp(min=1e-4)

        loss_mult = 1 / (2 * uncertainty.pow(2))
        loss_mult = loss_mult.clamp(max=1e4)

        cosine_part = (1 - cosine_feature.sub(0.5).div(0.5)).clip(0.0, 1).unsqueeze(1)
        cosine_loss = cosine_part * loss_mult

        valid_pixels = cosine_mask.sum()
        if valid_pixels > 0:
            cosine_loss_mean = (cosine_loss * cosine_mask).sum() / valid_pixels
        else:
            cosine_loss_mean = torch.tensor(0.0, device=uncertainty.device, requires_grad=True)

        if valid_pixels > 0:
            log_uncertainty_masked = torch.log(uncertainty + 1e-8)
            valid_log_uncertainty = log_uncertainty_masked * cosine_mask
            beta = valid_log_uncertainty.sum() / valid_pixels
            beta = beta.clamp(min=-5.0)
        else:
            beta = torch.tensor(0.0, device=uncertainty.device, requires_grad=True)

        uncertainty_loss = cosine_loss_mean + self.uncertainty_regularizer_weight * beta

        if torch.isnan(uncertainty_loss) or torch.isinf(uncertainty_loss):
            uncertainty_loss = torch.tensor(0.0, device=uncertainty.device, requires_grad=True)

        losses = {'loss_uncertainty': uncertainty_loss}
        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'pose': self.loss_pose,
            'uncertainty': self.loss_uncert,
        }
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets, is_aux=False):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        if ('aux_outputs' in outputs) and is_aux:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


def build(cfg):
    backbone = build_backbone(cfg)
    fp_transformer = build_fp_transformer(cfg)

    model = VOTransformer(
        backbone,
        fp_transformer,
        with_pose_refine=cfg['with_pose_refine'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        is_Semantics=cfg['is_Semantics'],
        DINOv3_version=cfg['DINOv3_version'],
        DINOv3_weights_dir=cfg['DINOv3_weights_dir'])

    weight_dict = {'loss_trans': cfg['trans_loss_coef'], 'loss_rot': cfg['rot_loss_coef'],
                   'loss_uncertainty': cfg['loss_uncertainty']}
    weight_layers = [0.2, 0.3, 0.4, 0.6, 0.8, 1]

    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': weight_layers[i] * v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': weight_layers[-1] * v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['pose', 'uncertainty']

    criterion = SetCriterion(
        weight_dict=weight_dict,
        losses=losses,
        uncertainty_regularizer_weight=cfg['uncertainty_regularizer_weight'])

    device = torch.device(cfg['device'])
    criterion.to(device)

    return model, criterion
