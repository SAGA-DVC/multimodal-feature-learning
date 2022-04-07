""" Modules for DVC """

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



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_1=0., dropout_2=0.):

        """
        Multi-layer perceptron which consists of 2 fully connected layers.
  
        Parameters:
            `in_dim` (int): Input dimension of the MLP block
            `hidden_dim` (int): Dimension of the intermediate layer
            `out_dim` (int): Output dimension of the MLP block
            `drouput_1` (float): Dropout probability applied after the first fully connected layer in the MLP block (default 0.0)
            `drouput_2` (float): Dropout probability applied after the second fully connected layer in the MLP block (default 0.0)
            
        """

        super(MLP, self).__init__()

        self.fully_connected_1 = nn.Linear(in_dim, hidden_dim)
        self.activation_layer = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_1)
        self.fully_connected_2 = nn.Linear(hidden_dim, out_dim)
        self.dropout_2 = nn.Dropout(dropout_2)

    def forward(self, x):

        """
        Performs a forward pass on the Multi-layer perceptron.

        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        x = self.fully_connected_1(x) # (batch_size, num_tokens, hidden_dim)
        x = self.activation_layer(x)
        x = self.dropout_1(x) 
        x = self.fully_connected_2(x)  # (batch_size, num_tokens, out_dim)
        x = self.dropout_2(x) 

        return x

class FFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.):

        """
        Feed Forward Network with 'n' layers
  
        Parameters:
            `in_dim` (int): Input dimension of the MLP block
            `hidden_dim` (int): Dimension of the intermediate layer
            `out_dim` (int): Output dimension of the MLP block
            `num_layers` (int): Depth of FFN
            `drouput` (float): Dropout probability applied after the first fully connected layer in the MLP block (default 0.0)
            
        """

        super(FFN, self).__init__()

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim]))
        self.relu = nn.ReLU()

    def forward(self, x):

        """
        Performs a forward pass on the Feed Forward Network.
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        """

        for i, layer in enumerate(self.layers):
            x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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