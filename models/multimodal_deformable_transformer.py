import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from .modules.misc_modules import inverse_sigmoid 
from .modules.attention import MSDeformAttn

class MultimodalDeformableTransformer(nn.Module):
    '''Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=256).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=1024).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer(default=relu)
        return_intermediate_dec: whether to return intermediate outputs of decoder(default=false)
        num_feature_levels: number of feature levels in multiscale Deformable Attention (default=4)
        dec_n_points: number of sampling points per attention head per feature level for decoder (default=4)
        enc_n_points: number of sampling points per attention head per feature level for encoder (default=4)
    '''
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.no_encoder = (num_encoder_layers == 0)
        self.num_feature_levels = num_feature_levels

        encoder_layer = MultimodalDeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = MultimodalDeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = MultimodalDeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = MultimodalDeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.pos_trans = nn.Linear(d_model, d_model * 2)
        self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        self.reference_points = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 256
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, num_token, 2
        proposals = proposals.sigmoid() * scale
        # N, num_token, 2, 256
        pos = proposals[:, :, :, None] / dim_t
        # N, num_token, 2, 128, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def get_valid_ratio(self, mask):
        valid_ratio_L = torch.sum(~mask, 1).float() / mask.shape[1]
        return valid_ratio_L
      
    def prepare_encoder_inputs(self, srcs, masks, pos_embeds):
        '''
        :param srcs (list[[batch_size, d_model, num_tokens]])
        :param masks (list[[batch_size, num_tokens]])
        :param pos_embeds (list[[batch_size, d_model, num_tokens]])

        :return src_flatten (batch_size, sum of num_tokens in all level, d_model)
        :return temporal_shapes (num_feature_levels)    #   list of num token at each level
        :return level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        :return valid_ratios (batch_size, num_feature_levels)
        :return lvl_pos_embed_flatten (batch_size, sum of num_token in all level, d_model)
        :return mask_flatten (batch_size, sum of num_tokens in all level)
        '''
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            """
            src: (batch_size, d_model, num_token)
            mask: (batch_size, num_token)
            pos_embed: (batch_size, d_model, num_token)
            """
            batch_size, d_model, num_token = src.shape
            temporal_shapes.append(num_token)
            src = src.transpose(1, 2)  #    (batch_size, num_token, d_model)
            pos_embed = pos_embed.transpose(1, 2)  #    (batch_size, num_token, d_model)

            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    #(batch_size, num_token, d_model)   
            
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1) #   (batch_size, sum of num_tokens in all level, d_model)
        mask_flatten = torch.cat(mask_flatten, 1)   #   (batch_size, sum of num_tokens in all level)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)   #   (batch_size, sum of num_token in all level, d_model)
        
        temporal_shapes = torch.as_tensor(temporal_shapes, dtype=torch.long, device=src_flatten.device) #   list of num token at each level
        level_start_index = torch.cat((temporal_shapes.new_zeros((1,)), temporal_shapes.cumsum(0)[:-1]))    #   list to find the start index of each level from flatten tensor 
        
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)   #   (batch_size, num_feature_levels)

        return src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten

    def forward_encoder(self, video_src_flatten, video_temporal_shapes, video_level_start_index, video_valid_ratios, video_lvl_pos_embed_flatten,
                        video_mask_flatten, audio_src_flatten, audio_temporal_shapes, audio_level_start_index, audio_valid_ratios, audio_lvl_pos_embed_flatten,
                        audio_mask_flatten):
        """
            :param video_src_flatten (batch_size, sum of num_token in all level, d_model)
            :param video_temporal_shapes: (num_feature_levels)    #   list of num token at each level
            :param video_level_start_index: (num_feature_levels)  #   list to find the start index of each level from flatten tensor
            :param video_valid_ratios: (batch_size, num_feature_levels)
            :param video_lvl_pos_embed_flatten: (batch_size, sum of num_token in all level, d_model)
            :param video_mask_flatten: (batch_size, sum of num_tokens in all level)
            :param audio_src_flatten (batch_size, sum of num_token in all level, d_model)
            :param audio_temporal_shapes: (num_feature_levels)    #   list of num token at each level
            :param audio_level_start_index: (num_feature_levels)  #   list to find the start index of each level from flatten tensor
            :param audio_valid_ratios: (batch_size, num_feature_levels)
            :param audio_lvl_pos_embed_flatten: (batch_size, sum of num_token in all level, d_model)
            :param audio_mask_flatten: (batch_size, sum of num_tokens in all level)


            return: memory -> (audio_attended_visual, visual_attended_audio)
            audio_attended_visual: (batch_size, sum of num_token in all level, d_model) 
            visual_attended_audio: (batch_size, sum of num_token in all level, d_model)     
        """
        # encoder
        if self.no_encoder:
            memory = video_src_flatten, audio_src_flatten 
        else:
            memory = self.encoder(video_src_flatten, video_temporal_shapes, video_level_start_index, video_valid_ratios, video_lvl_pos_embed_flatten,
                        video_mask_flatten, audio_src_flatten, audio_temporal_shapes, audio_level_start_index, audio_valid_ratios, audio_lvl_pos_embed_flatten,
                        audio_mask_flatten)

        return memory

    def prepare_decoder_input_query(self, batch_size, query_embed):
        '''
        param: memory (batch_size, sum of num_token in all level, d_model)
        param: query_embed (num_queries, d_model * 2)

        return: init_reference_out (batch_size, num_queries, 1)
        return: tgt (batch_size, num_queries, d_model)
        return: reference_points (batch_size, num_queries, 1)
        return: query_embed (num_queries, d_model * 2)
        '''
        
        query_embed, tgt = torch.chunk(query_embed, 2, dim=1)   #   tgt->(num_queries, d_model)  query_embed->(num_queries, d_model)
       
        query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)   #   (batch_size, num_queries, d_model)  
        tgt = tgt.unsqueeze(0).expand(batch_size, -1, -1)   #   (batch_size, num_queries, d_model) 
        reference_points = self.reference_points(query_embed).sigmoid() #   nn.Linear(d_model, 1)  shape-> (batch_size, num_queries, 1)  
        init_reference_out = reference_points   #   (batch_size, num_queries, 1)
        
        return init_reference_out, tgt, reference_points, query_embed

    def prepare_decoder_input_proposal(self, gt_reference_points):
        topk_coords_unact = inverse_sigmoid(gt_reference_points)
        reference_points = gt_reference_points
        init_reference_out = reference_points
        pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_embed, tgt = torch.chunk(pos_trans_out, 2, dim=2)
        return init_reference_out, tgt, reference_points, query_embed

    def forward_decoder(self, *kargs):
        hs, inter_references_out = self.decoder(*kargs)
        return hs, inter_references_out


class MultimodalDeformableTransformerEncoderLayer(nn.Module):
    '''Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=256).
        d_ffn: the dimension of the feedforward network model (default=1024).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer(default=relu).
        n_levels: number of feature levels in multiscale Deformable Attention (default=4).
        n_heads: the number of heads in the multiheadattention models (default=8).
        n_points: number of sampling points per attention head per feature level for encoder (default=4)    
    '''
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, video_src, video_pos, video_reference_points, video_temporal_shapes, video_level_start_index, video_padding_mask, audio_src, audio_pos, audio_reference_points, audio_temporal_shapes, audio_level_start_index, audio_padding_mask):
        '''
        param: video_src (batch_size, sum of num_token in all level, d_model)
        param: video_pos (batch_size, sum of num_token in all level, d_model) #  lvl_pos_embed_flatten
        param: video_reference_points (batch_size, sum of num_token in all level, num_feature_levels, 1)
        param: video_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: video_level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: video_padding_mask (batch_size, sum of num_tokens in all level)
        param: audio_src (batch_size, sum of num_token in all level, d_model)
        param: audio_pos (batch_size, sum of num_token in all level, d_model) #  lvl_pos_embed_flatten
        param: audio_reference_points (batch_size, sum of num_token in all level, num_feature_levels, 1)
        param: audio_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: audio_level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: audio_padding_mask (batch_size, sum of num_tokens in all level)
        
        return: audio_attended_visual: (batch_size, sum of num_token in all level, d_model) 
        return: visual_attended_audio: (batch_size, sum of num_token in all level, d_model) 
        '''
        #   self attention for video
        video_src2 = self.self_attn(self.with_pos_embed(video_src, video_pos), video_reference_points, video_src, video_temporal_shapes, video_level_start_index,
                              video_padding_mask)
        video_src = video_src + self.dropout1(video_src2)
        video_src = self.norm1(video_src)

        #   self attention for audio
        audio_src2 = self.self_attn(self.with_pos_embed(audio_src, audio_pos), audio_reference_points, audio_src, audio_temporal_shapes, audio_level_start_index,
                              audio_padding_mask)
        audio_src = audio_src + self.dropout1(audio_src2)
        audio_src = self.norm1(audio_src)

        #   multimodal attention
        visual_attended_audio = self.self_attn(audio_src, audio_reference_points, video_src, video_temporal_shapes, video_level_start_index,
                              video_padding_mask)
        audio_attended_visual = self.self_attn(video_src, video_reference_points, audio_src, audio_temporal_shapes, audio_level_start_index,
                              audio_padding_mask)

        # ffn
        visual_attended_audio = self.forward_ffn(visual_attended_audio)
        audio_attended_visual = self.forward_ffn(audio_attended_visual)

        return audio_attended_visual, visual_attended_audio, 


class MultimodalDeformableTransformerEncoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(temporal_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (L_) in enumerate(temporal_shapes):
            ref = torch.linspace(0.5, L_ - 0.5, L_, dtype=torch.float32, device=device) #   Creates a one-dimensional tensor of size 3rd param whose values are evenly spaced from 1st param to 2nd param
            ref = ref.reshape(-1)[None] / (valid_ratios[:, None, lvl] * L_)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        reference_points = reference_points[:,:,:,None]
        return reference_points

    def forward(self, video_src, video_temporal_shapes, video_level_start_index, video_valid_ratios, video_pos, video_padding_mask, audio_src, audio_temporal_shapes, audio_level_start_index, audio_valid_ratios, audio_pos, audio_padding_mask):
        """
        param: video_src (batch_size, sum of num_token in all level, d_model)
        param: video_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: video_level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: video_valid_ratios (batch_size, num_feature_levels)
        param: video_pos (batch_size, sum of num_token in all level, d_model) #  lvl_pos_embed_flatten
        param: video_padding_mask (batch_size, sum of num_tokens in all level)
        param: audio_src (batch_size, sum of num_token in all level, d_model)
        param: audio_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: audio_level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: audio_valid_ratios (batch_size, num_feature_levels)
        param: audio_pos (batch_size, sum of num_token in all level, d_model) #  lvl_pos_embed_flatten
        param: audio_padding_mask (batch_size, sum of num_tokens in all level)

        return: memory -> (audio_attended_visual, visual_attended_audio)
        audio_attended_visual: (batch_size, sum of num_token in all level, d_model) 
        visual_attended_audio: (batch_size, sum of num_token in all level, d_model) 

        """
        
        output = video_src, audio_src
        video_reference_points = self.get_reference_points(video_temporal_shapes, video_valid_ratios, device=video_src.device)  # (batch_size, sum of num_token in all level, num_feature_levels, 1)
        audio_reference_points = self.get_reference_points(audio_temporal_shapes, audio_valid_ratios, device=audio_src.device)  # (batch_size, sum of num_token in all level, num_feature_levels, 1)
        
        for _, layer in enumerate(self.layers):
            video_src, audio_src = output
            output = layer(video_src, video_pos, video_reference_points, video_temporal_shapes, video_level_start_index, video_padding_mask, audio_src, audio_pos, audio_reference_points, audio_temporal_shapes, audio_level_start_index, audio_padding_mask)
        return output


class MultimodalDeformableTransformerDecoderLayer(nn.Module):
    '''Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=256).
        d_ffn: the dimension of the feedforward network model (default=1024).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer(default=relu).
        n_levels: number of feature levels in multiscale Deformable Attention (default=4).
        n_heads: the number of heads in the multiheadattention models (default=8).
        n_points: number of sampling points per attention head per feature level for decoder (default=4)    
    '''
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # bridge
        self.norm4 = nn.LayerNorm(2*d_model)
        self.linear3 = nn.Linear(2*d_model, d_model)
        self.dropout5 = nn.Dropout(dropout)
        

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, query_mask, video_src, video_temporal_shapes, video_level_start_index,
                video_src_padding_mask, audio_src, audio_temporal_shapes, audio_level_start_index,
                audio_src_padding_mask):

        """
        param: tgt (batch_size, num_queries, d_model)
        param: query_pos (num_queries, hidden_dim * 2)
        param: reference_points (batch_size, num_queries, 1)
        param: query_mask (batch_size, num_queries)

        param: video_src (batch_size, sum of num_token in all level, d_model)
        param: video_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: video_level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor    
        param: video_padding_mask (batch_size, sum of num_tokens in all level)
        param: audio_src (batch_size, sum of num_token in all level, d_model)
        param: audio_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: audio_level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor    
        param: audio_padding_mask (batch_size, sum of num_tokens in all level)
        

        return: output (batch_size, num_queries, d_model)
        """
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=~query_mask)[
            0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention for video
        tgt_video = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               video_src, video_temporal_shapes, video_level_start_index, video_src_padding_mask)
        tgt_video = tgt + self.dropout1(tgt_video)
        tgt_video = self.norm1(tgt_video)

        # cross attention for video
        tgt_audio = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               audio_src, audio_temporal_shapes, audio_level_start_index, audio_src_padding_mask)
        tgt_audio = tgt + self.dropout1(tgt_audio)
        tgt_audio = self.norm1(tgt_audio)

        # bridge
        tgt = torch.cat([tgt_video, tgt_audio], dim=-1) #   (batch_size, num_queries, 2*d_model)
        tgt = self.norm4(tgt)
        tgt = self.linear3(tgt) #   (batch_size, num_queries, d_model)
        tgt = self.dropout5(tgt)
        tgt = self.activation(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class MultimodalDeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_head = None

    def forward(self, tgt, reference_points, query_pos, query_padding_mask, video_src, video_temporal_shapes, video_level_start_index, video_valid_ratios,
                video_padding_mask, audio_src, audio_temporal_shapes, audio_level_start_index, audio_valid_ratios,
                audio_padding_mask,  disable_iterative_refine=False):
        """
        param: tgt (batch_size, num_queries, d_model)
        param: reference_points (batch_size, num_queries, 1)
        param: query_pos (num_queries, hidden_dim * 2)
        param: query_padding_mask (batch_size, num_queries)

        param: video_src (batch_size, sum of num_token in all level, d_model)
        param: video_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: video_level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: video_valid_ratios (batch_size, num_feature_levels)
        param: video_padding_mask (batch_size, sum of num_tokens in all level)
        param: audio_src (batch_size, sum of num_token in all level, d_model)
        param: audio_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: audio_level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: audio_valid_ratios (batch_size, num_feature_levels)
        param: audio_padding_mask (batch_size, sum of num_tokens in all level)
        
        param: disable_iterative_refine bool
        
        return: output: (number of decoder_layers, batch_size, num_queries, d_model)
        return: reference_points: (number of decoder_layers, batch_size, num_queries, 1)
        """
        output = tgt    #   (batch_size, num_queries, d_model)

        intermediate = []
        intermediate_reference_points = []
        bs = tgt.shape[0]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.stack([video_valid_ratios, video_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 1
                reference_points_input = reference_points[:, :, None] * video_valid_ratios[:, None, :, None]  #   (batch_size, num_queries, num_feature_levels, 1)
            output = layer(output, query_pos, reference_points_input, query_padding_mask, video_src, video_temporal_shapes, video_level_start_index, video_padding_mask, audio_src, audio_temporal_shapes, audio_level_start_index, audio_padding_mask)    #   (batch_size, num_queries, d_model)
            
            # hack implementation for iterative bounding box refinement
            if disable_iterative_refine:
                reference_points = reference_points
            else:
                if (self.bbox_head is not None):
                    tmp = self.bbox_head[lid](output)
                    if reference_points.shape[-1] == 2:
                        new_reference_points = tmp + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 1
                        new_reference_points = tmp
                        new_reference_points[..., :1] = tmp[..., :1] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()
                else:
                    reference_points = reference_points

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}."
)


def build_multimodal_deformable_transformer(args):
    return MultimodalDeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.transformer_ff_dim,
        dropout=args.transformer_dropout_prob,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points)