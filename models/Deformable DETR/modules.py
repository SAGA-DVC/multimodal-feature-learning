from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import math
import warnings
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from torch.autograd import Function

from deformable_transformer import DeformableTransformer
try:
    import MultiScaleDeformableAttention as MSDA
except:
    pass
from torch.autograd.function import once_differentiable
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        # sampling_locations:(...,2), the first item of last dim means x axis corresponding to w, and second item of the last dim means y, corresponding to h.
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights, return_value=False):
    # for debug and test only,
    # need to use cuda version instead
    '''
    :param value (batch_size, sum of num_token in all level, nhead, dmodel/nhead)  
    :param value_spatial_shapes (num_feature_levels, 2)
    :param sampling_locations (batch_size, sum of num_token in all level, n_heads, n_levels, n_points, 2) 
    :param attention_weights (batch_size, sum of num_token in all level, n_heads, n_levels, n_points)
    :param return_value
    '''
    N_, S_, M_, D_ = value.shape    # N_: batch size , S_: \sum_H*W, M_ : head number, D_: feature dim of each head

    _, Lq_, M_, L_, P_, _ = sampling_locations.shape  # Lq_: \sum H*W, L_: multi-scale number, P_: number of sampled key points

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    sampling_grids = 2 * sampling_locations - 1 # convert value from range[0,1] to [-1, 1]
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # sampling_grid_l_: (...,2), the first item of last dim means x axis corresponding to w, and second item of the last dim means y, corresponding to h.
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='border', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)

    if return_value:
        return torch.stack(sampling_value_list, dim=-2)
    #(N_ * M_, D_, Lq_, L_* P_) * (N_*M_, 1, Lq_, L_*P_) --> (N_*M_, D_, Lq_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()



def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points )
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid_init = grid_init[..., 0].repeat(1, self.n_levels, self.n_points)
        for i in range(self.n_points):
            grid_init[:, :, i] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 1), range in [0, 1], including padding area
                                        or (N, Length_{query}, n_levels, 2), add additional (c, l) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} T_l, C)
        :param input_spatial_shapes        (n_levels ), [T_0, T_1, ..., T_{L-1}]
        :param input_level_start_index     (n_levels ), [0, 1_0, T_0+T_1, ...]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :return output                     (N, Length_{query}, C)
        """
        
        N, Len_q, _ = query.shape   #   (batch_size, sum of num_token in all level, dmodel)
        N, Len_in, _ = input_flatten.shape  #   (batch_size, sum of num_token in all level, dmodel)
        assert input_spatial_shapes.sum() == Len_in

        value = self.value_proj(input_flatten)  # linear transformation --> nn.Linear(d_model, d_model) shape->(batch_size, sum of num_token in all level, dmodel)
        
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)   #   (batch_size, sum of num_token in all level, nhead, dmodel/nhead)    

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)  # linear transformation --> nn.Linear(d_model, n_heads * n_levels * n_points ) shape->(batch_size, sum of num_token in all level, n_heads, n_levels, n_points)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)   #   linear transformation --> nn.Linear(d_model, n_heads * n_levels * n_points) shape->(batch_size, sum of num_token in all level, n_heads, n_levels*n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points) #   (batch_size, sum of num_token in all level, n_heads, n_levels, n_points)


        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 1:
            offset_normalizer = input_spatial_shapes
            sampling_locations = reference_points[:, :, None, :, None, 0] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None]  #   (batch_size, sum of num_token in all level, n_heads, n_levels, n_points)                   
        elif reference_points.shape[-1] == 2:
            sampling_locations = reference_points[:, :, None, :, None, 0] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 1] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 1 or 2, but get {} instead.'.format(reference_points.shape[-1]))

        if True:
            sampling_locations = torch.stack((sampling_locations, 0.5 * sampling_locations.new_ones(sampling_locations.shape)), -1)  #   (batch_size, sum of num_token in all level, n_heads, n_levels, n_points, 2) 
            input_spatial_shapes = torch.stack([input_spatial_shapes.new_ones(input_spatial_shapes.shape), input_spatial_shapes], -1)   # (num_feature_levels, 2)
            
        if query.device.type == 'cuda':
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
                self.im2col_step)
        else:
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor], duration=None):
        self.tensors = tensors
        self.mask = mask
        self.duration = duration

    def to(self, device, non_blocking=False):
        '''# type: (Device) -> NestedTensor # noqa'''

        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
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


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.max_duration = 256
        self.duration_embed_layer = nn.Linear(self.max_duration, self.max_duration)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        duration = tensor_list.duration
        assert mask is not None
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            x_embed = (x_embed - 0.5) / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

        dur_embed = self.duration_embedding(duration).reshape(-1,1,self.max_duration).expand_as(pos_x)
        pos = torch.cat((pos_x, dur_embed), dim=2).permute(0, 2, 1)
        return pos

    def duration_embedding(self, durations):
        out = torch.zeros(len(durations), self.max_duration, device=durations.device)
        durations = durations.int()
        for ii in range(len(durations)):
            out[ii, :durations[ii]] = 1
        out = self.duration_embed_layer(out)
        return out


def build_position_encoding(position_embedding, N_steps):
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return 
    

def decide_two_stage(transformer_input_type, gt_boxes, gt_boxes_mask, criterion):
    if transformer_input_type == 'gt_proposals':
        two_stage = True
        proposals = gt_boxes
        proposals_mask = gt_boxes_mask
        criterion.matcher.cost_caption = 0
        for q_k in ['loss_length', 'loss_ce', 'loss_bbox', 'loss_giou']:
            for key in criterion.weight_dict.keys():
                if q_k in key:
                    criterion.weight_dict[key] = 0
        disable_iterative_refine = True
    elif transformer_input_type == 'queries':  #
        two_stage = False
        proposals = None
        proposals_mask = None
        disable_iterative_refine = False
    else:
        raise ValueError('Wrong value of transformer_input_type, got {}'.format(transformer_input_type))
    return two_stage, disable_iterative_refine, proposals, proposals_mask
