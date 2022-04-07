""" Different types of embedding layers """

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



class TokenEmbedding(nn.Module):
    def __init__(self, img_size=224, spatial_patch_size=16, temporal_patch_size=1, in_channels=3, d_model=768, layer_norm=None):
        
        """
        Converts video into token embeddings based on specified patches that convolve over the video. Based on the
        temporal patch size, these embeddings can follow 'uniform frame sampling' or 'tubelet embedding' schemes.

        Parameters:
            `img_size` (int): dimension of one frame of the video (should be a square i.e. height=width) (default 224)
            `spatial_patch_size` (int): dimension of the patch that will be used to convolve over a frame (default 16)
            `temporal_patch_size` (int): dimension of the patch that will be used to convolve over multiple frames (default 1)
            `in_channels` (int): number of channels of the each frame in the video. e.g. RGB. (default 3)
            `num_classes` (int): number of classes for the prediction task (default 1000)
            `d_model` (int): Dimension of the tensors used to compute attention

        """
        
        super(TokenEmbedding, self).__init__()

        self.temporal_patch_size = temporal_patch_size

        self.project_to_patch_embeddings = nn.Conv3d(in_channels, d_model, 
                                                    kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size), 
                                                    stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size))
        self.layer_norm = layer_norm(d_model) if layer_norm else nn.Identity()

    def forward(self, x):

        """
        3D Convolutions are used to get the token embeddings. 
  
        Parameters:
           x (tensor): Tensor of dimension (batch_size, in_channels, num_frames, img_size, img_size)

        Returns:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        """

        x = self.project_to_patch_embeddings(x) # (batch_size, d_model, num_frames, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(3)  # (batch_size, d_model, num_frames, num_patches)
        x = x.permute(0, 2, 3, 1)  # (batch_size, num_frames, num_patches, d_model)
        x = self.layer_norm(x)
        
        return x

# TODO - pos emebd to accomomdate variable length input
class PositionalEmbedding(nn.Module):
        
    def __init__(self, shape, positional_embedding_dropout=0.):

        """
        Positional embeddings are initialzed and added to the input tensors. 
  
        Parameters:
            `shape` (tuple of ints): Shape of the Positional Embedding
            `positional_embedding_dropout` (float): dropout probability for the positional embeddings (default 0.0)
          
        """

        super(PositionalEmbedding, self).__init__()

        self.pos_shape = shape
        self.positional_embedding = nn.Parameter(torch.zeros(*shape)) 
        self.positional_embedding_dropout = nn.Dropout(p=positional_embedding_dropout)
    
    def forward(self, x):

        """
        Adds positional embeddings to the input tensors. 
  
        Parameters:
           x (tensor): Tensor of dimension (batch_size, num_tokens, d_model) OR (batch_size, num_frames, num_patches, d_model)

        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model) OR (batch_size, num_frames, num_patches, d_model)

        """

        _, num_tokens, d_model = x.shape
        
        assert self.pos_shape[1] >= num_tokens, f"Dimension 1 (num_tokens) of positional embedding {self.pos_shape[1]} set at initialisation time should be >= the dimension 1 (num_tokens) of input {num_tokens}."
        assert self.pos_shape[2] == d_model, f"Dimension 2 (d_model) of positional embedding {self.pos_shape[2]} set at initialisation time does not match the dimension 2 (d_model) of input {d_model}."
        
        x = self.positional_embedding_dropout(x + self.positional_embedding[:, :num_tokens]) 

        return x


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
        self.max_duration = 384
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
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
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

        

class VocabularyEmbedder(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(VocabularyEmbedder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # replaced if pretrained weights are used
        self.embedder = nn.Embedding(vocab_size, d_model)

    def forward(self, x):  # x - tokens (B, seq_len)
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        return x  # (B, seq_len, d_model)

    def init_word_embeddings(self, embedding_matrix, emb_weights_req_grad=True):
        if embedding_matrix is None:
            print('Training word embeddings from scratch')
        else:
            _, pretrained_embed_dim = embedding_matrix.shape
            if self.d_model == pretrained_embed_dim:
                self.embedder = self.embedder.from_pretrained(embedding_matrix)
                self.embedder.weight.requires_grad = emb_weights_req_grad
                print(f'Glove embedding dimension is of the same size as d_model i.e. {self.d_model}')
            else:
                self.embedder = nn.Sequential(
                    nn.Embedding(self.vocab_size, pretrained_embed_dim).from_pretrained(embedding_matrix),
                    nn.Linear(pretrained_embed_dim, self.d_model),
                    nn.ReLU()
                )
                self.embedder[0].weight.requires_grad = emb_weights_req_grad
                print(f'Glove embedding dimension is {pretrained_embed_dim} and d_model is {self.d_model}')