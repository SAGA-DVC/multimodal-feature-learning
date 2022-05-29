""" Attention Modules """

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from typing import Optional
import math
import warnings

import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from torch.autograd import Function

try:
    import MultiScaleDeformableAttention as MSDA
except:
    pass
from torch.autograd.function import once_differentiable
# needed due to empty tensor bug in pytorch and torchvision 0.5



class Attention(nn.Module):

    def __init__(self, d_model, num_heads=12, qkv_bias=False, attention_dropout=0., projection_dropout=0., init=''):

        """
        Initialises all the attributes of the for the multi-headed attention block. 
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. (default 12)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
        """

        super(Attention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model == self.head_dim * num_heads, "The model dimension must be divisible by the number of heads."

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.projection_layer = nn.Linear(d_model, d_model)
        # self.projection_dropout = nn.Dropout(projection_dropout)

    # add q, k ,v
    def forward(self, x, attn_mask=None, key_padding_mask=None, need_weights=False):

        """
        Performs a forward pass on the multi-headed attention block followed by a linear (projection) layer.
  
        Parameters:
            x (Tensor): Tensor of dimension (batch_size, num_tokens, d_model)
            mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens)
        
        Returns:
            x (Tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        batch_size, num_tokens, d_model = x.shape
        
        qkv = self.qkv(x) # (batch_size, num_tokens, dim * 3)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, num_tokens, head_dim)

        query, key, value = qkv.unbind(0) 
        
        # (batch_size, num_heads, num_tokens, head_dim) * (batch_size, num_heads, head_dim, num_tokens) 
        # -> (batch_size, num_heads, num_tokens, num_tokens)
        self_attention = torch.matmul(query, key.transpose(-2, -1))

        # if mask is not None:
        #     self_attention = self_attention.masked_fill(mask, float("-1e20"))

        if attn_mask is not None:
            self_attention = self_attention.masked_fill(attn_mask, float("-1e20"))
        
        if key_padding_mask is not None:
            self_attention = self_attention.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), float("-1e20"))
            
        self_attention = (self_attention * self.scale).softmax(dim=-1)
        self_attention = self.attention_dropout(self_attention)

        weighted_attention = torch.matmul(self_attention, value) # (batch_size, num_heads, num_tokens, head_dim)
        weighted_attention = weighted_attention.transpose(1, 2).flatten(2) # (batch_size, num_tokens, d_model)
        
        x = self.projection_layer(weighted_attention) # (batch_size, num_tokens, d_model)
        # x = self.projection_dropout(x)

        if need_weights:
            return x, self_attention
        else:
            return x, None
        

# TODO - add mask
class DotProductAttention(nn.Module):
    def __init__(self, d_model, num_heads=12, qkv_bias=False, attention_dropout=0., projection_dropout=0.):

        """
        Initialises all the attributes for the Dot Product Attention architecture. 
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. (default 12)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
          
        """

        super(DotProductAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        assert d_model == self.head_dim * self.num_heads, "The model dimension must be divisible by the number of heads."

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.projection_layer = nn.Linear(d_model, d_model)
        self.projection_dropout = nn.Dropout(projection_dropout)

    # masks not yet added
    def forward(self, x):

        """
        Performs a forward pass on the Dot Product Attention block which fuses the spatial and temporal attention outputs.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        """

        batch_size, num_frames, num_patches, d_model = x.shape
        x = x.reshape(batch_size, -1, d_model) # (batch_size, num_frames * num_patches, d_model)
        
        qkv = self.qkv(x) # (batch_size, num_frames * num_patches, d_model * 3)
        qkv = qkv.reshape(batch_size, num_frames * num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, num_frames * num_patches, head_dim)

        query, key, value = qkv.unbind(0) # (batch_size, num_heads, num_frames * num_patches, head_dim)

        query_spatial, query_temporal = query.chunk(2, dim=1) # (batch_size, num_heads/2, num_frames * num_patches, head_dim)
        key_spatial, key_temporal = key.chunk(2, dim=1) # (batch_size, num_heads/2, num_frames * num_patches, head_dim)
        value_spatial, value_temporal = value.chunk(2, dim=1) # (batch_size, num_heads/2, num_frames * num_patches, head_dim)

        query_spatial = query_spatial.reshape(batch_size * num_frames, self.num_heads // 2, 
                                            num_patches, -1)
        key_spatial = key_spatial.reshape(batch_size * num_frames, self.num_heads // 2, 
                                            num_patches, -1)
        value_spatial = value_spatial.reshape(batch_size * num_frames, self.num_heads // 2, 
                                            num_patches, -1)

        query_temporal = query_temporal.reshape(batch_size * num_patches, self.num_heads // 2, 
                                            num_frames, -1)
        key_temporal = key_temporal.reshape(batch_size * num_patches, self.num_heads // 2, 
                                            num_frames, -1)
        value_temporal = value_temporal.reshape(batch_size * num_patches, self.num_heads // 2, 
                                            num_frames, -1)
        
        # (batch_size * num_frames, num_heads/2, num_patches, head_dim) * (batch_size * num_frames, num_heads/2, head_dim, num_patches) 
        # -> (batch_size * num_frames, num_heads/2, num_patches, num_patches)
        self_attention_spatial = torch.matmul(query_spatial, key_spatial.transpose(-2, -1)) * self.scale
        self_attention_spatial = self_attention_spatial.softmax(dim=-1)
        self_attention_spatial = self.attention_dropout(self_attention_spatial)

        weighted_attention_spatial = torch.matmul(self_attention_spatial, value_spatial) # (batch_size * num_frames, num_heads/2, num_patches, head_dim)
        
        # (batch_size * num_patches, num_heads/2, num_frames, head_dim) * (batch_size * num_patches, num_heads/2, head_dim, num_frames) 
        # -> (batch_size * num_patches, num_heads/2, num_frames, num_frames)
        self_attention_temporal = torch.matmul(query_temporal, key_temporal.transpose(-2, -1)) * self.scale
        self_attention_temporal = self_attention_temporal.softmax(dim=-1)
        self_attention_temporal = self.attention_dropout(self_attention_temporal)

        weighted_attention_temporal = torch.matmul(self_attention_temporal, value_temporal) # (batch_size * num_patches, num_heads/2, num_frames , head_dim)

        weighted_attention_spatial = weighted_attention_spatial.reshape(batch_size, self.num_heads // 2, num_frames * num_patches, -1)
        weighted_attention_temporal = weighted_attention_temporal.reshape(batch_size, self.num_heads // 2, num_frames * num_patches, -1)

        weighted_attention = torch.cat((weighted_attention_spatial, weighted_attention_temporal), dim=1) # (batch_size, num_heads, num_frames * num_patches, head_dim)

        weighted_attention = weighted_attention.transpose(1, 2).flatten(2) # (batch_size, num_frames * num_patches, d_model)
        
        x = self.projection_layer(weighted_attention) # (batch_size, num_frames * num_patches, d_model)
        x = self.projection_dropout(x)

        x = x.reshape(batch_size, num_frames, num_patches, d_model)

        return x


class CrossAttention(nn.Module):

    def __init__(self, d_model, num_heads=12, qkv_bias=False, attention_dropout=0., projection_dropout=0.):

        """
        Initialises all the attributes of the for the cross attention block which involves different modalities (eg. video and audio). 
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. (default 12)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
        """

        super(CrossAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model == self.head_dim * num_heads, "The model dimension must be divisible by the number of heads."

        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=qkv_bias)

        self.attention_dropout = nn.Dropout(attention_dropout)

        self.projection_layer = nn.Linear(d_model, d_model)
        # self.projection_dropout = nn.Dropout(projection_dropout)


    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, need_weights=False):

        """
        Performs a forward pass on the cross attention block followed by a linear (projection) layer.
  
        Parameters:
            q (tensor): Tensor of dimension (batch_size, num_tokens_q, d_model) represeting a query vector
            k (tensor): Tensor of dimension (batch_size, num_tokens_k, d_model) represeting a key vector
            v (tensor): Tensor of dimension (batch_size, num_tokens_v, d_model) represeting a value vector
            mask (tensor) : Tensor of dimension (batch_size, 1, 1, num_tokens_k) represeting a value vector

        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens_q, d_model)

        """

        assert k.shape == v.shape, f"The keys and values inputted to the cross attention module should have the same shape. However, key has {k.shape} and value has {v.shape}."

        batch_size, num_tokens_q, _ = q.shape
        _, num_tokens_k, _ = k.shape
        
        q = self.q_linear(q) # (batch_size, num_tokens_q, d_model)
        k = self.k_linear(k) # (batch_size, num_tokens_k, d_model)
        v = self.v_linear(v) # (batch_size, num_tokens_k, d_model)

        q = q.reshape(batch_size, num_tokens_q, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3) # (batch_size, num_heads, num_tokens_q, head_dim)
        
        k = k.reshape(batch_size, num_tokens_k, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3) # (batch_size, num_heads, num_tokens_k, head_dim)

        v = v.reshape(batch_size, num_tokens_k, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3) # (batch_size, num_heads, num_tokens_k, head_dim)
        
        # (batch_size, num_heads, num_tokens_q, head_dim) * (batch_size, num_heads, head_dim, num_tokens_k) 
        # -> (batch_size, num_heads, num_tokens_q, num_tokens_k)
        cross_attention = torch.matmul(q, k.transpose(-2, -1))
        
        # if mask is not None:
        #     cross_attention = cross_attention.masked_fill(mask, float("-1e20"))

        if attn_mask is not None:
            cross_attention = cross_attention.masked_fill(attn_mask, float("-1e20"))
        
        if key_padding_mask is not None:
            cross_attention = cross_attention.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), float("-1e20"))

        cross_attention = (cross_attention * self.scale).softmax(dim=-1)
        cross_attention = self.attention_dropout(cross_attention)

        weighted_cross_attention = torch.matmul(cross_attention, v) # (batch_size, num_heads, num_tokens_q, head_dim)
        weighted_cross_attention = weighted_cross_attention.transpose(1, 2).flatten(2) # (batch_size, num_tokens_q, d_model)
        
        x = self.projection_layer(weighted_cross_attention) # (batch_size, num_tokens_q, d_model)
        # x = self.projection_dropout(x)

        if need_weights:
            return x, cross_attention
        else:
            return x, None



class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_temporal_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        # sampling_locations:(...,2), the first item of last dim means x axis corresponding to w, and second item of the last dim means y, corresponding to h.
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_temporal_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_temporal_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_temporal_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_temporal_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_temporal_shapes, sampling_locations, attention_weights, return_value=False):
    # for debug and test only,
    # need to use cuda version instead
    '''
    :param value (batch_size, sum of num_token in all level, nhead, dmodel/nhead)  
    :param value_temporal_shapes (num_feature_levels, 1)
    :param sampling_locations (batch_size, sparse_tokens, n_heads, n_levels, n_points, 1) 
    :param attention_weights (batch_size, sparse_tokens, n_heads, n_levels, n_points)
    :param return_value
    '''

    batch_size, num_tokens, num_heads, d_model_per_head = value.shape    # batch_size: batch size , num_tokens: \sum_H*W, num_heads : head number, d_model_per_head: feature dim of each head

    _, num_sparse_tokens, num_heads, num_feature_levels, num_points, _ = sampling_locations.shape  # num_sparse_tokens: sparse_tokens, num_feature_levels: multi-scale number, num_points: number of sampled key points

    value_list = value.split([num_feature_levels for num_feature_levels in value_temporal_shapes], dim=1)   # [(B, 400, nhead, dmodel/nhead), (B, 200, nhead, dmodel/nhead), ...]

    # (batch_size, sparse_tokens, n_heads, n_levels, n_points, 1)
    sampling_grids = 2 * sampling_locations - 1 # convert value from range[0,1] to [-1, 1]

    sampling_value_list = []
    for lid_, (current_lvl_tokens) in enumerate(value_temporal_shapes):
        # batch_size, num_feature_levels, num_heads, d_model_per_head -> batch_size, num_feature_levels, num_heads*d_model_per_head -> batch_size, num_heads*d_model_per_head, num_feature_levels-> batch_size*num_heads, d_model_per_head, num_feature_levels, 1
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(batch_size*num_heads, d_model_per_head, current_lvl_tokens).unsqueeze(-1)
        
        # batch_size, num_sparse_tokens, num_heads, num_points, 1 -> batch_size, num_heads, num_sparse_tokens, num_points, 1 -> batch_size*num_heads, num_sparse_tokens, num_points, 1
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)

        # batch_size*num_heads, num_sparse_tokens, num_points, 1 -> batch_size*num_heads, num_sparse_tokens * num_points, 1 -> batch_size*num_heads, 1, num_sparse_tokens * num_points
        sampling_grid_l_ = sampling_grid_l_.flatten(1, 2).reshape(sampling_grid_l_.shape[0], 1, -1)

        # batch_size*num_heads, 1, num_sparse_tokens * num_points, 2
        sampling_grid_l_ = torch.stack([-torch.ones_like(sampling_grid_l_), sampling_grid_l_], dim=-1)

        # sampling_grid_l_: (...,2), the first item of last dim is '-1', and second item of the last dim means t_start.
        # batch_size*num_heads, d_model_per_head, 1, num_sparse_tokens * num_points
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='border', align_corners=False)

        sampling_value_l_ = sampling_value_l_.reshape(batch_size*num_heads, d_model_per_head, num_sparse_tokens, num_points)
        sampling_value_list.append(sampling_value_l_)

    # (batch_size, num_sparse_tokens, num_heads, num_feature_levels, num_points) -> (batch_size, num_heads, num_sparse_tokens, num_feature_levels, num_points) -> (batch_size, num_heads, 1, num_sparse_tokens, num_feature_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(batch_size*num_heads, 1, num_sparse_tokens, num_feature_levels*num_points)

    if return_value:
        print("[UNUSED] RETURN VALUE in attention.py: ", return_value)
        return torch.stack(sampling_value_list, dim=-2)
    
    # (batch_size * num_heads, d_model_per_head, num_sparse_tokens, num_feature_levels* num_points) * (batch_size*num_heads, 1, num_sparse_tokens, num_feature_levels*num_points) --> (batch_size*num_heads, d_model_per_head, num_sparse_tokens)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(batch_size, num_heads*d_model_per_head, num_sparse_tokens)

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
        # constant_(self.attention_weights.weight.data, 0.)
        xavier_uniform_(self.attention_weights.weight.data)    # changed
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, is_sparse=False):
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
            # value = value.masked_fill(input_padding_mask[..., None] == False, float(0))     # changed (for inverted mask)
        
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

        # if True:
        #     sampling_locations = torch.stack((sampling_locations, 0.5 * sampling_locations.new_ones(sampling_locations.shape)), -1)  #   (batch_size, sum of num_token in all level, n_heads, n_levels, n_points, 2) 
        #     input_spatial_shapes = torch.stack([input_spatial_shapes.new_ones(input_spatial_shapes.shape), input_spatial_shapes], -1)   # (num_feature_levels, 2)

        # if query.device.type == 'cuda':
        #     output = MSDeformAttnFunction.apply(
        #         value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
        #         self.im2col_step)
        # else:

        input_spatial_shapes = input_spatial_shapes.unsqueeze(-1)
        sampling_locations = sampling_locations.unsqueeze(-1)

        # print(':='*40)
        # print("value: ", value.shape)
        # print("input_spatial_shapes: ", input_spatial_shapes)
        # print("sampling_locations: ", sampling_locations.shape)
        # print("attention_weights: ", attention_weights.shape)
        # print(':='*40)

        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        if(is_sparse):
            return output, sampling_locations, attention_weights
        else:
            return output