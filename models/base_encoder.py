"""
Base Encoder to create multi-level conv features and positional embedding.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .modules.misc_modules import NestedTensor
from .modules.embedding_layers import PositionEmbeddingSine, PositionEmbeddingVideoSine


class BaseEncoder(nn.Module):
    '''Args:
        num_feature_levels: number of feature levels in multiscale Deformable Attention (default=4)
        vf_dim: dim of frame-level feature vector (default = 500)
        d_model: Dimensionality of the hidden layer in the feed-forward networks within the Transformer (default=512).
    '''
    def __init__(self, num_feature_levels, vf_dim, d_model):
        super(BaseEncoder, self).__init__()

        # TODO - check pos_embed
        self.pos_embed_video = PositionEmbeddingVideoSine(d_model//2, normalize=True)
        self.pos_embed_audio = PositionEmbeddingSine(d_model, normalize=True)

        self.num_feature_levels = num_feature_levels
        self.d_model = d_model

        #   creating base_encoder
        if num_feature_levels > 1:
            input_proj_list = []
            in_channels = vf_dim
            input_proj_list.append(nn.Sequential(
                nn.Conv1d(in_channels, d_model, kernel_size=1),
                nn.GroupNorm(32, d_model),
            ))
            for _ in range(num_feature_levels - 1):
                input_proj_list.append(nn.Sequential(
                    nn.Conv1d(in_channels, d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, d_model),
                ))
                in_channels = d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(vf_dim, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, vf, mask, duration, modality):
        """
        :param vf: video tensor, expected shape: (batch_size, num_tokens, d_model)
        :param mask: video mask, expected shape: (batch_size, num_tokens)
        :param duration: video length, expected shape: (batch_size)
        :return: srcs - list (len=num_feature_levels) - [(batch_size, d_model, num_tokens*)]
                        where num_tokens* depends on num_feature_levels (essentially gets halved for each level)
        :return: masks - list (len=num_feature_levels - [(batch_size, num_tokens)]
        :return: pos - list (len=num_feature_levels) - [(batch_size, d_model, num_tokens)]
        """
        vf = vf.transpose(1, 2)  # (batch_size, num_tokens, d_model) --> (batch_size, d_model, num_tokens)
        vf_nt = NestedTensor(vf, mask, duration)

        if modality == 'video':
            pos0 = self.pos_embed_video(vf_nt) 
        else:
            pos0 = self.pos_embed_audio(vf, mask) 
        
        srcs = []
        masks = []
        poses = []

        src0, mask0 = vf_nt.decompose() #   return vf,mask
        srcs.append(self.input_proj[0](src0))
        masks.append(mask0)
        poses.append(pos0)
        assert mask is not None

        for l in range(1, self.num_feature_levels):
            if l == 1:
                src = self.input_proj[l](vf_nt.tensors)
            else:
                src = self.input_proj[l](srcs[-1])
            m = vf_nt.mask
            mask = F.interpolate(m[None].float(), size=src.shape[-1:]).to(torch.bool)[0] #  upsample the mask to given shape

            if modality == 'video':
                pos_l = self.pos_embed_video(NestedTensor(src, mask, duration)).to(src.dtype)
            else:
                pos_l = self.pos_embed_audio(src, mask).to(src.dtype)

            srcs.append(src)
            masks.append(mask)
            poses.append(pos_l)
        return srcs, masks, poses

def build_base_encoder(args):
    base_encoder = BaseEncoder(args.num_feature_levels, args.feature_dim, args.d_model)
    return base_encoder