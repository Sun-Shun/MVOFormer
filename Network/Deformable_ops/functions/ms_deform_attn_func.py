# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step

        # ── AMP 兼容：CUDA 核只支持 FP32，FP16/BF16 统一在此转换 ──────────
        input_dtype = value.dtype
        if input_dtype != torch.float32:
            value              = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights  = attention_weights.float()
        # ──────────────────────────────────────────────────────────────────

        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, ctx.im2col_step)

        # 保存的张量已是 FP32，backward 直接使用
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index,
                              sampling_locations, attention_weights)
        ctx.input_dtype = input_dtype

        # 输出还原为原始精度，与模型其余部分保持一致
        return output.to(input_dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (value, value_spatial_shapes, value_level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors

        # grad_output 可能是 FP16/BF16，backward CUDA 核同样只支持 FP32
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights,
                grad_output.float(),
                ctx.im2col_step)

        # 梯度还原为原始精度，保证 AMP scaler 正常工作
        return (grad_value.to(ctx.input_dtype), None, None,
                grad_sampling_loc.to(ctx.input_dtype),
                grad_attn_weight.to(ctx.input_dtype), None)


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()