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
import torchaudio
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


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def predict_event_num(counter, query_features):
    # (batch_size, num_queries, d_model)
    query_features_pool = torch.max(query_features, dim=1, keepdim=False)[0]  # [batch_size, d_model]
    outputs_class0 = counter(query_features_pool)    # [batch_size, max_eseq_length + 1]
    return outputs_class0

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



def aframes_to_fbank(aframes: torch.Tensor, sample_frequency, num_mel_bins, target_length, ):
    aframes = aframes - aframes.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        aframes, 
        htk_compat=True,
        sample_frequency=sample_frequency, 
        use_energy=False,
        window_type='hanning', 
        num_mel_bins=num_mel_bins, 
        dither=0.0, 
        frame_shift=10)
    
    n_frames = fbank.shape[0]
    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    
    return fbank