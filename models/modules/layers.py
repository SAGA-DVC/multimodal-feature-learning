""" Encoder and Decoder layers """

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

from .attention import *



# TODO- dropout before/after each layer_norm?
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, dropout_1=0., dropout_2=0., 
                attention_dropout=0., projection_dropout=0., pre_norm=True):

        """
        EncoderLayer consisting of the basic attention architecture.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. 
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
        """

        super(EncoderLayer, self).__init__()

        self.pre_norm = pre_norm

        #eps for compatibility with ViT pretrained weights??
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6) 

        self.attention = Attention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    def forward(self, x):

        """
        Performs a forward pass on the Factorised Encoder block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        if self.pre_norm:
            return self.forward_pre(x)
        
        else:
            return self.forward_post(x)


    def forward_pre(self, x):

        """
        Performs a forward pass with pre-norm on the Factorised Encoder block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        x = x + self.attention(self.layer_norm_1(x)) # (batch_size, num_tokens, d_model)
        x = x + self.mlp(self.layer_norm_2(x)) # (batch_size, num_tokens, d_model)

        return x
    
    
    def forward_post(self, x):

        """
        Performs a forward pass with post-norm on the Factorised Encoder block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_tokens, d_model)

        """

        x = self.layer_norm_1(x + self.attentio(x)) # (batch_size, num_tokens, d_model)
        x = self.layer_norm_2(x + self.mlp(x)) # (batch_size, num_tokens, d_model)

        return x


# TODO - forward pre-post
class FactorisedSelfAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, 
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0.):
        
        """
        Attention architecture consisting of spatial attention followed by temporal attention within one block.
    
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
    
        """

        super(FactorisedSelfAttentionEncoderLayer, self).__init__()

        #eps for compatibility with ViT pretrained weights??
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.spatial_attention = Attention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        self.temporal_attention = Attention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    def forward(self, x):

        """
        Performs a forward pass on the Factorised Self-Attention block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        """

        batch_size, num_frames, num_patches, d_model = x.shape
        
        x = x.reshape(-1, num_patches, d_model) # (batch_size * num_frames, num_patches, d_model)

        x = x + self.spatial_attention(self.layer_norm_1(x)) # (batch_size * num_frames, num_patches, d_model)

        x = x.reshape(-1, num_frames, d_model) # (batch_size * num_patches, num_frames, d_model)

        x = x + self.temporal_attention(self.layer_norm_2(x)) # (batch_size * num_patches, num_frames, d_model)

        x = x + self.mlp(self.layer_norm_3(x)) # (batch_size * num_patches, num_frames, d_model)

        x = x.reshape(batch_size, num_frames, num_patches, d_model) # (batch_size, num_frames, num_patches, d_model)

        return x


# TODO - forward pre-post
class FactorisedDotProductAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False, 
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0.):
        
        """
        Attention architecture consisting of spatial attention fused with temporal attention within one block.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `depth` (int): number of encoder blocks. 
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
    
        """

        super(FactorisedDotProductAttentionEncoderLayer, self).__init__()
        
        #eps for compatibility with ViT pretrained weights??
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6) 

        self.attention = DotProductAttention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)
                                   
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    def forward(self, x):

        """
        Performs a forward pass on the Factorised Dot Product Attention block.
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)
        
        Returns:
            x (tensor): Tensor of dimension (batch_size, num_frames, num_patches, d_model)

        """

        batch_size, num_frames, num_patches, d_model = x.shape

        x = x + self.attention(self.layer_norm_1(x)) # (batch_size, num_frames, num_patches, d_model)

        x = x.reshape(batch_size, -1, d_model)

        x = x + self.mlp(self.layer_norm_2(x)) # (batch_size, num_frames * num_patches, d_model)

        x = x.reshape(batch_size, num_frames, num_patches, d_model)

        return x



# TODO- dropout before/after each layer_norm?
# TODO- check forward_pre sequence for self_attention (which one of q, k, v should have layer_norm?)
class BiModalEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True):

        """
        Bi-modal encoder block for cross attention b/w video and audio features.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads.
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
    
        """
        
        super(BiModalEncoderLayer, self).__init__()

        self.pre_norm = pre_norm

        self.attention_av = CrossAttention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.attention_va = CrossAttention(d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_av_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_va_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_av_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_va_2 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)

        self.mlp_av = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

        self.mlp_va = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)



    def forward(self, vid, aud):

        """
        Performs a forward pass over the Bi-modal encoder with video and audio features.
  
        Parameters:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features
        
        Returns:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features after cross attention with audio features 
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features after cross attention with video features 

        """
        
        if self.pre_norm:
            return self.forward_pre(vid, aud)
        
        else:
            return self.forward_post(vid, aud)
        
    

    def forward_pre(self, vid, aud):

        """
        Performs a forward pass over the Bi-modal encoder with video and audio features and normalisation layers before attention and mlp blocks.
  
        Parameters:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features
        
        Returns:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features after cross attention with audio features 
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features after cross attention with video features 

        """

        vid_after_norm = self.layer_norm_av_1(vid)
        aud_after_norm = self.layer_norm_va_1(aud)

        vid = vid + self.attention_av(vid_after_norm, aud_after_norm, aud_after_norm) # (batch_size, num_tokens, d_model)
        aud = aud + self.attention_va(aud_after_norm, vid_after_norm, vid_after_norm) # (batch_size, num_tokens, d_model)

        vid = vid + self.mlp_av(self.layer_norm_av_2(vid)) # (batch_size, num_tokens, d_model)
        aud = aud + self.mlp_va(self.layer_norm_va_2(aud)) # (batch_size, num_tokens, d_model)

        return vid, aud

    
    def forward_post(self, vid, aud):

        """
        Performs a forward pass over the Bi-modal encoder with video and audio features and normalisation layers after attention and mlp blocks.
  
        Parameters:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features
        
        Returns:
            vid (tensor): Tensor of dimension (batch_size, num_frames, d_model) representing video features after cross attention with audio features 
            aud (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing audio features after cross attention with video features 

        """

        vid = self.layer_norm_av_1(vid + self.attention_av(vid, aud, aud)) # (batch_size, num_tokens, d_model)
        aud = self.layer_norm_va_1(aud + self.attention_va(aud, vid, vid)) # (batch_size, num_tokens, d_model)

        vid = self.layer_norm_av_2(vid + self.mlp_av(vid)) # (batch_size, num_tokens, d_model)
        aud = self.layer_norm_va_2(aud + self.mlp_va(aud)) # (batch_size, num_tokens, d_model)

        return vid, aud


# TODO - dropout before/after each layer_norm?
# TODO- check forward_pre sequence for self_attention (which one of q, k, v should have layer_norm?)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,  
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True):

        """
        Decoder consisting of the basic attention architecture.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. 
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
        """

        super(DecoderLayer, self).__init__()

        self.pre_norm=pre_norm
        
        self.self_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.cross_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    
    def forward(self, target, memory, positional_embedding_layer, query_embedding, mask):

        """
        Performs a forward pass on the Decoder block. Calls either forward_pre() or forward_post() based on the value of self.pre_nrom
  
        Parameters:
            target (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding_layer: position embedding layer for encoder inputs
            query_embedding: event queries
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """

        if self.pre_norm:
            return self.forward_pre(target, memory, positional_embedding_layer, query_embedding, mask) # (batch_size, num_queries, d_model)
        else:
            return self.forward_post(target, memory, positional_embedding_layer, query_embedding, mask) # (batch_size, num_queries, d_model)

    
    def forward_pre(self, target, memory, positional_embedding_layer, query_embedding, mask):
        
        """
        Performs a forward pass on the Decoder block with normalisation layers before attention and mlp blocks.
  
        Parameters:
            target (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding_layer: position embedding layer for encoder inputs
            query_embedding: event queries
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """

        target_after_norm = self.layer_norm_1(target) 
        q = k = target_after_norm + query_embedding
        target = target + self.self_attention(q=q, k=k, v=target_after_norm, mask=mask) # (batch_size, num_queries, d_model)

        target_after_norm = self.layer_norm_2(target)
        q = target_after_norm + query_embedding
        k = positional_embedding_layer(memory)
        target = target + self.cross_attention(q=q, k=k, v=memory) # (batch_size, num_queries, d_model)
        
        target_after_norm = self.layer_norm_3(target)
        target = target + self.mlp(target_after_norm)

        return target


    def forward_post(self, target, memory, positional_embedding_layer, query_embedding, mask):

        """
        Performs a forward pass on the Decoder block with normalisation layers after attention and mlp blocks.
  
        Parameters:
            target (tensor): the sequence to the decoder layer, Tensor of dimension (batch_size, num_queries, d_model)
            memory: the sequence from the last layer of the encoder
            positional_embedding_layer: position embedding layer for encoder inputs
            query_embedding: event queries
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, num_queries, d_model)
        """
       
        q = k = target + query_embedding
        target = self.layer_norm_1(target + self.self_attention(q=q, k=k, v=target, mask=mask)) # (batch_size, num_queries, d_model)

        q = target + query_embedding
        k = positional_embedding_layer(memory)
        target = self.layer_norm_2(target + self.cross_attention(q=q, k=k, v=memory)) # (batch_size, num_queries, d_model)

        target = target + self.mlp(target)
        target = self.layer_norm_3(target)

        return target



# --------------------------------------------------------------
# Taken from https://github.com/v-iashin/BMT/blob/master/model/blocks.py


# TODO - dropout before/after each layer_norm?
# TODO - check forward_pre sequence for self_attention (which one of q, k, v should have layer_norm?)
# TODO - check pos embed for encoder -- how??
class CaptionDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4., qkv_bias=False,  
                attention_dropout=0., projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True):

        """
        Decoder consisting of the basic attention architecture.
  
        Parameters:
            `d_model` (int): Dimension of the tensors used to compute attention
            `num_heads` (int): Number of attention heads. 
            `mlp_ratio` (int): Used to determine the hidden layer dimension of the MLP. (default 4)
            `qkv_bias` (boolean): Determines whether to use bias as part of the query/key/value linear layers in the attention block (default True)
            `attention_dropout` (float): Dropout probability for the layer after the multi-head attention mechanism (default 0.0)
            `projection_dropout` (float): Dropout probability for the layer after the projection layer (default 0.0)
            `dropout_1` (float): Dropout probability for the MLP block (default 0.0)
            `dropout_2` (float): Dropout probability for the MLP block (default 0.0)
            `pre_norm` (boolean): If True, the normalisation layer would be placed before the attention and mlp blocks. Else, after them. (default True)
        """

        super(CaptionDecoderLayer, self).__init__()

        self.pre_norm=pre_norm
        
        self.self_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.cross_attention = CrossAttention(d_model=d_model, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attention_dropout=attention_dropout, projection_dropout=projection_dropout)

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, out_dim=d_model, 
                       dropout_1=dropout_1, dropout_2=dropout_2)

    
    def forward(self, target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask):

        """
        Performs a forward pass on the Decoder block. Calls either forward_pre() or forward_post() based on the value of self.pre_nrom
  
        Parameters:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size). The sequence to the decoder layer. 
            memory: Tensor of dimension (batch_size, num_tokens, d_model). The sequence from the last layer of the encoder
            word_positional_embedding_layer (nn.Module): position embedding layer for captions
            positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Target mask for the captions to be used in the self attention block
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size)
        """

        if self.pre_norm:
            return self.forward_pre(target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask) # (batch_size, num_queries, d_model)
        else:
            return self.forward_post(target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask) # (batch_size, num_queries, d_model)

    
    def forward_pre(self, target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask):
        
        """
        Performs a forward pass on the Decoder block with normalisation layers before attention and mlp blocks.
  
        Parameters:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size). The sequence to the decoder layer. 
            memory: Tensor of dimension (batch_size, num_tokens, d_model). The sequence from the last layer of the encoder
            word_positional_embedding_layer (nn.Module): position embedding layer for captions
            positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Target mask for the captions to be used in the self attention block
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size)
        """

        target_after_norm = self.layer_norm_1(target) 
        q = k = word_positional_embedding_layer(target_after_norm)
        target = target + self.self_attention(q=q, k=k, v=target_after_norm, mask=tgt_mask) # (batch_size, num_queries, d_model)

        target_after_norm = self.layer_norm_2(target)
        q = word_positional_embedding_layer(target_after_norm)
        k = positional_embedding_layer(memory)
        target = target + self.cross_attention(q=q, k=k, v=memory, mask=memory_mask) # (batch_size, num_queries, d_model)
        
        target_after_norm = self.layer_norm_3(target)
        target = target + self.mlp(target_after_norm)

        return target


    def forward_post(self, target, memory, word_positional_embedding_layer, positional_embedding_layer, tgt_mask, memory_mask):

        """
        Performs a forward pass on the Decoder block with normalisation layers after attention and mlp blocks.
  
        Parameters:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size). The sequence to the decoder layer. 
            memory: Tensor of dimension (batch_size, num_tokens, d_model). The sequence from the last layer of the encoder
            word_positional_embedding_layer (nn.Module): position embedding layer for captions
            positional_embedding_layer (nn.Module): position embedding layer for encoder inputs
            tgt_mask (Tensor): Tensor of dimension (batch_size, 1, seq_len, seq_len). Target mask for the captions to be used in the self attention block
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens). Memory padding mask to be used in the cross attention block
        
        Returns:
            target (tensor): Tensor of dimension (batch_size, seq_len, vocab_size)
        """
       
        q = k = word_positional_embedding_layer(target)
        target = self.layer_norm_1(target + self.self_attention(q=q, k=k, v=target, mask=matgt_masksk)) # (batch_size, num_queries, d_model)

        q = word_positional_embedding_layer(target)
        k = positional_embedding_layer(memory)
        target = self.layer_norm_2(target + self.cross_attention(q=q, k=k, v=memory, mask=memory_mask)) # (batch_size, num_queries, d_model)

        target = target + self.mlp(target)
        target = self.layer_norm_3(target)

        return target

