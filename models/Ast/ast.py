import torch
import torch.nn as nn
import os
from modules import VisionTransformer
os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.models.layers import to_2tuple,trunc_normal_

class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, model_size='base384', model_official=None):
        
        """
        The Audio Spectrogram Transformer (AST) model.
        :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
        :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
        :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
        :param input_fdim: the number of frequency bins of the input spectrogram
        :param input_tdim: the number of time frames of the input spectrogram
        :param imagenet_pretrain: if use ImageNet pretrained model
        :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
        :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
        `model_official`: This model's weights are used by AST
        """
        super(AudioSpectrogramTransformer, self).__init__()
 
        
        self.model_official = model_official
        self.encoder = VisionTransformer(d_model=768, num_heads=12)
        self.original_num_patches = self.model_official.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.model_official.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.encoder.patch_embeddings_layer.num_patches = num_patches
      

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain == True:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.model_official.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.model_official.patch_embed.proj.bias
        self.encoder.patch_embeddings_layer.project_to_patch_embeddings = new_proj

        # the positional embedding
        if imagenet_pretrain == True:
            # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
            new_pos_embed = self.model_official.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)


            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            
            # flatten the positional embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)


            # concatenate the above positional embedding with the cls token and distillation token of the deit model.
            self.encoder.positional_embedding = nn.Parameter(torch.cat([self.model_official.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        else:
            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.model_official.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.encoder.positional_embedding = new_pos_embed
            trunc_normal_(self.model_official.pos_embed, std=.02)



    # helper function to get intermediate shape
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

 
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (10, 100, 128)
        x = x.unsqueeze(1) #(batch_size, time_frame_num, frequency_bins) -> (batch_size, in_channels = 1, time_frame_num, frequency_bins)
        x = x.transpose(2, 3) #(batch_size, in_channels = 1, time_frame_num, frequency_bins) -> (batch_size, in_channels = 1, frequency_bins, time_frame_num)
        x = self.encoder(x) #(batch_size, in_channels = 1, frequency_bins, time_frame_num) -> (batch_size, d_model)
        x = self.mlp_head(x) # (batch_size, d_model) -> (batch_size, class)
        return x
