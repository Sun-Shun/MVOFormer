import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import math
import copy
from Network.Deformable_ops.modules import MSDeformAttn, MSDeformAttn_cross, MultiheadAttention
from Tool.Utils.utils import inverse_sigmoid
from Network.Model.attention import MultiHeadAttentionWeighted, MultiHeadAttention, FeedForwardNet
from Network.Model.temporal_conv import TemporalEncoder


class MLP(nn.Module):
    """Simple multi-layer perceptron (FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class FlowEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src,
                                 spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class FlowEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None,
                padding_mask=None, ref_token_index=None, ref_token_coord=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class UncertaintyModule(nn.Module):
    def __init__(self, d_model=256, ctx_dim=384,
                 activation="relu", n_heads=8, dropout=0.1,
                 d_ffn=1024, qkv_bias=True, qk_rms_norm_cross=False):
        super().__init__()

        self.cross_attn_uncert = MultiHeadAttention(ctx_dim,
                                                    ctx_channels=d_model,
                                                    num_heads=n_heads,
                                                    type="cross",
                                                    attn_mode="full",
                                                    qkv_bias=qkv_bias,
                                                    qk_rms_norm=qk_rms_norm_cross)
        self.norm_uncert = nn.LayerNorm(ctx_dim)

        self.ffn = FeedForwardNet(ctx_dim)
        self.norm_ffn = nn.LayerNorm(ctx_dim)

        self.temporal = TemporalEncoder(num_bev_queue=2, embed_dims=ctx_dim, num_block=1)
        self.mask = nn.Conv2d(ctx_dim, 1, kernel_size=1)
        self.uncertainty_dropout = nn.Dropout2d(p=0.5)
        self.bn = nn.BatchNorm2d(ctx_dim)
        nn.init.normal_(self.mask.weight.data, 0, 0.01)
        nn.init.zeros_(self.mask.bias)

        self.nhead = n_heads

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm_ffn(tgt)
        return tgt

    def _forward_uncertainty_features(self, x):
        x = self.uncertainty_dropout(x)
        x = self.bn(x)
        logits = self.mask(x)
        logits = logits + math.log(math.exp(1) - 1)
        logits = F.softplus(logits)
        logits = logits.clamp(min=0.1)
        return logits

    def forward(self, pose_kv, pose_kv_pos, depth_feature):
        bs, ctx_dim, h, w = depth_feature[0].shape
        feature = depth_feature[0]
        pre_feature = depth_feature[1]

        query = self.temporal(feature, pre_feature)
        x_query = query
        kv = self.with_pos_embed(pose_kv, pose_kv_pos)
        query_uncert = self.cross_attn_uncert(x_query, kv)
        query = x_query + query_uncert
        query = self.norm_uncert(query)

        query = self.forward_ffn(query)

        context = query.permute(0, 2, 1).reshape(bs, ctx_dim, h, w)
        uncertainty = self._forward_uncertainty_features(context)
        loss_mult = 1 / (2 * uncertainty.pow(2)).squeeze(1).reshape(bs, h * w)
        mask_mult = loss_mult.clamp_max(3)

        return query, uncertainty, mask_mult


class PoseDecoderLayer(nn.Module):
    def __init__(self, d_model=256, ctx_dim=384, d_ffn=1024, dropout=0.1,
                 activation="relu", n_levels=4, n_heads=8, n_points=4,
                 qkv_bias=True, qk_rms_norm=True, qk_rms_norm_cross=False):
        super().__init__()

        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn_feature = MultiHeadAttentionWeighted(d_model,
                                                             ctx_channels=ctx_dim,
                                                             num_heads=n_heads,
                                                             type="cross",
                                                             attn_mode="full",
                                                             qkv_bias=qkv_bias,
                                                             qk_rms_norm=qk_rms_norm_cross)
        self.norm_feature = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model,
                                            num_heads=n_heads,
                                            type="self",
                                            attn_mode='full',
                                            qkv_bias=qkv_bias,
                                            qk_rms_norm=qk_rms_norm)

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNet(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.nhead = n_heads
        self.d_model = d_model
        self.uncertainty_module = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points,
                src, src_spatial_shapes, level_start_index,
                src_padding_mask, bs, DINOv3_feature=None, is_first=False):

        x = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(x)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # Flow cross-attention
        tgt_flow, _ = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                       reference_points,
                                       src, src_spatial_shapes, level_start_index,
                                       src_padding_mask)

        # Uncertainty-guided cross-attention with DINOv3 features
        if self.uncertainty_module and DINOv3_feature is not None:
            kv_uncert, uncertainty, mask_mult = self.uncertainty_module(tgt, query_pos, DINOv3_feature)
            tgt2 = self.cross_attn_feature(self.with_pos_embed(tgt, query_pos), kv_uncert,
                                           mask_weight=mask_mult)
            tgt = tgt + tgt_flow + 0.8 * tgt2[0]
            tgt = self.norm_feature(tgt)
        else:
            uncertainty = None
            tgt = tgt + tgt_flow
            tgt = self.norm_feature(tgt)

        tgt = self.forward_ffn(tgt)

        return tgt, uncertainty


class PoseDecoder(nn.Module):
    def __init__(self, decoder_layer, uncertainty_module, num_layers, with_pose_refine=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.uncertainty_module = nn.ModuleList([uncertainty_module for _ in range(num_layers)])
        for i in range(len(self.layers)):
            self.layers[i].uncertainty_module = self.uncertainty_module[i]

        self.num_layers = num_layers
        self.with_pose_refine = with_pose_refine
        self.delta_ref_point = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
                src_valid_ratios, query_pos=None, src_padding_mask=None, bs=None,
                DINOv3_feature=None):
        output = tgt

        intermediate = []
        intermediate_uncertainty = []
        intermediate_reference_points = []

        bs = src.shape[0]

        for lid, layer in enumerate(self.layers):
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output, uncertainty = layer(
                output, query_pos, reference_points_input,
                src, src_spatial_shapes, src_level_start_index, src_padding_mask,
                bs, DINOv3_feature, is_first=(lid == 0))

            if self.with_pose_refine:
                tmp = self.delta_ref_point[lid](output)
                new_reference_points = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points

            intermediate.append(output)
            intermediate_uncertainty.append(uncertainty)
            intermediate_reference_points.append(reference_points)

        if uncertainty is None:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), None
        else:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), \
                torch.stack(intermediate_uncertainty)


class FPTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, activation="relu", with_pose_refine=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4, ctx_dim=384):
        super().__init__()

        self.ctx_dim = ctx_dim
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = FlowEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                         num_feature_levels, nhead, enc_n_points)
        self.encoder = FlowEncoder(encoder_layer, num_encoder_layers)

        if self.ctx_dim is not None:
            uncertainty_module = UncertaintyModule(d_model, self.ctx_dim, activation, nhead,
                                                   dropout, dim_feedforward)
        else:
            uncertainty_module = None

        decoder_layer = PoseDecoderLayer(d_model, self.ctx_dim, dim_feedforward, dropout, activation,
                                         num_feature_levels, nhead, dec_n_points)

        self.decoder = PoseDecoder(decoder_layer, uncertainty_module, num_decoder_layers,
                                   with_pose_refine)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, DINOv3_feature=None):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            mask = mask.flatten(1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                              lvl_pos_embed_flatten, mask_flatten)

        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()

        hs, inter_references, uncertainty = self.decoder(
            tgt, reference_points, memory, spatial_shapes,
            level_start_index, valid_ratios, query_embed,
            mask_flatten, bs=bs, DINOv3_feature=DINOv3_feature)

        return hs, reference_points, inter_references, uncertainty


def build_fp_transformer(cfg):
    return FPTransformer(d_model=cfg['hidden_dim'],
                         dropout=cfg['dropout'],
                         activation="gelu",
                         nhead=cfg['nheads'],
                         dim_feedforward=cfg['dim_feedforward'],
                         num_encoder_layers=cfg['enc_layers'],
                         num_decoder_layers=cfg['dec_layers'],
                         with_pose_refine=cfg['with_pose_refine'],
                         num_feature_levels=cfg['num_feature_levels'],
                         dec_n_points=cfg['dec_n_points'],
                         enc_n_points=cfg['enc_n_points'])
