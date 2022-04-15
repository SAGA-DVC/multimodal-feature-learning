"""
Audio Spectogram Transformer (AST)

Code used from the following repositories:
1. https://github.com/YuanGongND/ast
"""

import os

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from .modules.encoders import AstEncoder



class AudioSpectrogramTransformer(nn.Module):

    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                imagenet_pretrained=True, model_size='base384', model_official=None, 
                depth=12, d_model=768, num_heads=12, return_prelogits=False, return_preclassifier=False):
        
        """
        The Audio Spectrogram Transformer (AST) model.

        Parameters:
            `label_dim` : the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
            `fstride` : the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
            `tstride` : the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
            `input_fdim` : the number of frequency bins of the input spectrogram
            `input_tdim` : the number of time frames of the input spectrogram
            `imagenet_pretrained` : if use ImageNet pretrained model
            `audioset_pretrained` : if use full AudioSet and ImageNet pretrained model
            `model_size` : the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
            `model_official`: This model's weights are used by AST

        """
        super(AudioSpectrogramTransformer, self).__init__()
 
        
        self.model_official = model_official

        self.d_model = d_model

        self.encoder = AstEncoder(img_size=input_fdim, 
                        d_model=self.d_model, 
                        num_heads=num_heads, 
                        depth=depth, 
                        in_channels=1, 
                        num_classes=0)
        
        self.original_num_patches = self.model_official.patch_embed.num_patches
        self.original_hw = int(self.original_num_patches ** 0.5)

        self.original_embedding_dim = self.model_official.pos_embed.shape[2]

        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))


        # automatically get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.encoder.patch_embeddings_layer.num_patches = num_patches
      

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        
        if imagenet_pretrained == True:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.model_official.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.model_official.patch_embed.proj.bias
        
        self.encoder.patch_embeddings_layer.project_to_patch_embeddings = new_proj


        # the positional embedding
        if imagenet_pretrained == True:
            
            # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
            new_pos_embed = self.model_official.pos_embed[:, 1:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.original_hw, self.original_hw)

            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= self.original_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.original_hw / 2) - int(t_dim / 2): int(self.original_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.original_hw, t_dim), mode='bilinear')
            
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= self.original_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.original_hw / 2) - int(f_dim / 2): int(self.original_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            
            # flatten the positional embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)

            # concatenate the above positional embedding with the cls token and distillation token of the deit model.
            self.encoder.positional_embedding = nn.Parameter(torch.cat([self.model_official.pos_embed[:, :1, :].detach(), new_pos_embed], dim=1))
        
        else:
            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.model_official.patch_embed.num_patches + 1, self.original_embedding_dim))
            self.encoder.positional_embedding = new_pos_embed
            trunc_normal_(self.model_official.pos_embed, std=.02)
        
        self.return_prelogits = return_prelogits
        self.return_preclassifier = return_preclassifier



    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=64):

        """
        Helper function to get intermediate shape
        
        Parameters:
            `fstride` : the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
            `tstride` : the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
            `input_fdim` : the number of frequency bins of the input spectrogram
            `input_tdim` : the number of time frames of the input spectrogram
        
        Returns:
            `fdim` : the number of computed frequency bins of the input spectrogram
            `tdim` : the number of computed time frames of the input spectrogram
        """

        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        
        return f_dim, t_dim

 
    def forward(self, x):
        
        """
        Forward pass for the AST model

        Parameters:
            `x` (Tensor): the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        
        Returns:
            x (Tensor): if return_preclassifier is True, Tensor of dimension (batch_size, num_patches + 1, d_model)
                        if return_prelogits is True, Tensor of dimension (batch_size, d_model)
                        else Tensor of dimension (batch_size, num_classes)
        """

        x = x.unsqueeze(1)    # (batch_size, in_channels = 1, time_frame_num, frequency_bins)
        
        x = self.encoder(x)    # (batch_size, num_patches + 1, d_model)
        
        if self.return_preclassifier:
            return x
        
        x = x[:, 0]    # (batch_size, d_model)
        
        if self.return_prelogits:
            return x

        x = self.mlp_head(x)    # (batch_size, num_classes)
        
        return x


def build_ast(args):
    return AudioSpectrogramTransformer(**args)