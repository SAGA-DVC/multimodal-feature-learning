import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_
from ..modules.misc_modules import inverse_sigmoid, predict_event_num
from ..modules.attention import MSDeformAttn


class MultimodalSparseDeformableTransformer(nn.Module):
    '''Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=256).
        num_head: the number of heads in the multiheadattention models (default=8).
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
    def __init__(self, d_model=256, num_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4, rho=0.3, use_enc_aux_loss=False,
                 eff_query_init=False, eff_specific_head=False):
        super().__init__()

        self.d_model = d_model
        self.num_head = num_head

        self.no_encoder = (num_encoder_layers == 0)
        self.num_feature_levels = num_feature_levels
        self.eff_query_init = eff_query_init
        self.eff_specific_head = eff_specific_head
        self.rho = rho
        self.two_stage = False
        self.use_enc_aux_loss = use_enc_aux_loss
        self.sparse_enc_head = 1 if self.two_stage and self.rho else 0

        if self.rho:
            self.enc_mask_predictor = MaskPredictor(self.d_model, self.d_model)
        else:
            self.enc_mask_predictor = None

        encoder_layer = MultimodalSparseDeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, num_head, enc_n_points)
        self.encoder = MultimodalSparseDeformableTransformerEncoder(encoder_layer, num_encoder_layers, self.d_model)

        decoder_layer = MultimodalSparseDeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, num_head, dec_n_points)
        self.decoder = MultimodalSparseDeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # TODO: only when two_stage
        # self.pos_trans = nn.Linear(d_model, d_model * 2)
        # self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        # if self.two_stage:
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.reference_points = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _log_args(self, *names):
        print('==============')
        print("\n".join([f"{name}: {getattr(self, name)}" for name in names]))
        print('==============')

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

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, temporal_shapes, process_output=True):
        """Make region proposals for each multi-scale features considering their shapes and padding masks, 
        and project & normalize the encoder outputs corresponding to these proposals.
            - center points: relative grid coordinates in the range of [0.01, 0.99] (additional mask)
            - width/height:  2^(layer_id) * s (s=0.05) / see the appendix A.4
        
        param: memory (batch_size, sum of num_tokens in all levels, d_model)
        param: memory_padding_mask (batch_size, sum of num_tokens in all levels)
        param: temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: process_output bool

        return output_memory (batch_size, sum of num_tokens in all levels, d_model)
        return output_proposals (batch_size, sum of num_tokens in all levels, 2)
        return (~memory_padding_mask).sum(axis=-1) (batch_size)  ---valid_token_nums
        
        """
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, (L_) in enumerate(temporal_shapes):
            # level of encoded feature scale
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + L_)].view(N_, L_, 1)    #   (batch_size, num_token in L layer, 1)
            valid_L = torch.sum(~mask_flatten_[:, :, 0], 1)
            grid = torch.linspace(0, L_ - 1, L_, dtype=torch.float32, device=memory.device) #   (num_token in L layer)
            scale = valid_L.unsqueeze(-1)
            grid = (grid.unsqueeze(0).expand(N_, -1) + 0.5) / scale #   (batch_size, num_token in L layer)
            # wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            wh = torch.full(grid.shape, 0.05 * (2.0 ** lvl))    #   effiecint code for above line (batch_size, num_token in L layer)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 2)    #   (batch_size, num_token in L layer, 2) 2 is for centre_offset x length
            proposals.append(proposal)
            _cur += (L_)


        output_proposals = torch.cat(proposals, 1)  #   (batch_size, sum of num_tokens in all levels, 2)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)  #   (batch_size, sum of num_tokens in all levels, 1)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  #  inverse of sigmoid (batch_size, sum of num_tokens in all levels, 2)
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))    #   (batch_size, sum of num_tokens in all levels, 2)
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))  #   sigmoid(inf) = 1  (batch_size, sum of num_tokens in all levels, 2)
        output_memory = memory  #   (batch_size, sum of num_tokens in all levels, d_model)
        if process_output:
            output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))  #   (batch_size, sum of num_tokens in all levels, d_model)
            output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))    #   (batch_size, sum of num_tokens in all levels, d_model)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))    #   (batch_size, sum of num_tokens in all levels, d_model)
        # print("(~memory_padding_mask).sum(axis=-1)", (~memory_padding_mask).sum(axis=-1).shape)
        return output_memory, output_proposals, (~memory_padding_mask).sum(axis=-1)

    def get_valid_ratio(self, mask):
        valid_ratio_L = torch.sum(~mask, 1).float() / mask.shape[1]
        # valid_ratio_L = torch.sum(mask, 1).float() / mask.shape[1]    # changed (for inverted masks used in PDVC)
        return valid_ratio_L
      
    def prepare_encoder_inputs(self, srcs, masks, pos_embeds):
        '''
        :param srcs (list[[batch_size, d_model, num_tokens]])
        :param masks (list[[batch_size, num_tokens]])
        :param pos_embeds (list[[batch_size, d_model, num_tokens]])

        :return src_flatten (batch_size, sum of num_tokens in all levels, d_model)
        :return temporal_shapes (num_feature_levels)    #   list of num token at each level
        :return level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        :return valid_ratios (batch_size, num_feature_levels)
        :return lvl_pos_embed_flatten (batch_size, sum of num_token in all levels, d_model)
        :return mask_flatten (batch_size, sum of num_tokens in all levels)
        :return backbone_output_proposals (batch_size, sum of num_tokens in all levels, 2)
        :return backbone_topk_proposals (batch_size, backbone_topk)
        :return sparse_token_nums (batch_size)
        '''
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            """
            src: (batch_size, d_model, num_tokens)
            mask: (batch_size, num_token)
            pos_embed: (batch_size, d_model, num_tokens)
            """
            batch_size, d_model, num_tokens = src.shape
            temporal_shapes.append(num_tokens)
            src = src.transpose(1, 2)  #    (batch_size, num_tokens, d_model)
            pos_embed = pos_embed.transpose(1, 2)  #    (batch_size, num_tokens, d_model)
            
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    # (batch_size, num_tokens, d_model)   
            
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1) #   (batch_size, sum of num_tokens in all level, d_model)
        mask_flatten = torch.cat(mask_flatten, 1)   #   (batch_size, sum of num_tokens in all level)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)   #   (batch_size, sum of num_token in all level, d_model)
        
        temporal_shapes = torch.as_tensor(temporal_shapes, dtype=torch.long, device=src_flatten.device) #   list of num token at each level
        level_start_index = torch.cat((temporal_shapes.new_zeros((1,)), temporal_shapes.cumsum(0)[:-1]))    #   list to find the start index of each level from flatten tensor 
        
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)   #   (batch_size, num_feature_levels)

        # prepare for sparse encoder
        if self.rho or self.use_enc_aux_loss:
            #   backbone_output_memory <= (batch_size, sum of num_tokens in all levels, d_model)
            #   backbone_output_proposals <= (batch_size, sum of num_tokens in all levels, 2)
            #   valid_token_nums <= (batch_size)
            backbone_output_memory, backbone_output_proposals, valid_token_nums = self.gen_encoder_output_proposals(
                src_flatten+lvl_pos_embed_flatten, mask_flatten, temporal_shapes, 
                process_output=bool(self.rho))
            
            self.valid_token_nums = valid_token_nums

        if self.rho:
            sparse_token_nums = (valid_token_nums * self.rho).int() + 1
            backbone_topk = int(max(sparse_token_nums))
            self.sparse_token_nums = sparse_token_nums
            backbone_topk = min(backbone_topk, backbone_output_memory.shape[1])
            backbone_mask_prediction = self.enc_mask_predictor(backbone_output_memory).squeeze(-1)  #   (batch_size, sum of num_tokens in all levels)
            # excluding pad area
            backbone_mask_prediction = backbone_mask_prediction.masked_fill(mask_flatten, backbone_mask_prediction.min())   #   (batch_size, sum of num_tokens in all levels)
            backbone_topk_proposals = torch.topk(backbone_mask_prediction, backbone_topk, dim=1)[1] #   (batch_size, backbone_topk)
            
        else:
            backbone_topk_proposals = None
            backbone_outputs_class = None
            backbone_outputs_coord_unact = None
            backbone_mask_prediction = None
            sparse_token_nums= None

        output = {
            'src_flatten': src_flatten, 
            'temporal_shapes': temporal_shapes, 
            'level_start_index': level_start_index, 
            'valid_ratios': valid_ratios, 
            'lvl_pos_embed_flatten': lvl_pos_embed_flatten, 
            'mask_flatten': mask_flatten,  
            'backbone_output_proposals': backbone_output_proposals, 
            'backbone_topk_proposals': backbone_topk_proposals, 
            'backbone_mask_prediction': backbone_mask_prediction,
            'sparse_token_nums': sparse_token_nums
        }

        return output

    def forward_encoder(self, video_input, audio_input):
        """
            :param src_flatten (batch_size, sum of num_token in all level, d_model)
            :param temporal_shapes: (num_feature_levels)    #   list of num token at each level
            :param level_start_index: (num_feature_levels)  #   list to find the start index of each level from flatten tensor
            :param valid_ratios: (batch_size, num_feature_levels)
            :param lvl_pos_embed_flatten: (batch_size, sum of num_token in all level, d_model)
            :param mask_flatten: (batch_size, sum of num_tokens in all level)
            :return backbone_output_proposals (batch_size, sum of num_tokens in all levels, 2)
            :return backbone_topk_proposals (batch_size, backbone_topk)
            :return sparse_token_nums (batch_size)

            :return memory (batch_size, sum of num_token in all level, d_model) #   Multi-scale frame features
        """
       # encoder
        video_input['backbone_output_proposals'] = video_input['backbone_output_proposals'] if self.use_enc_aux_loss else None 
        audio_input['backbone_output_proposals'] = audio_input['backbone_output_proposals'] if self.use_enc_aux_loss else None 
        video_output, video_sampling_locations_enc, video_attn_weights_enc, audio_output, audio_sampling_locations_enc, audio_attn_weights_enc, video_enc_inter_outputs_class, video_enc_inter_outputs_count, video_enc_inter_outputs_coords, audio_enc_inter_outputs_class, audio_enc_inter_outputs_count, audio_enc_inter_outputs_coords = self.encoder(video_input, audio_input)

        return video_output, video_sampling_locations_enc, video_attn_weights_enc, audio_output, audio_sampling_locations_enc, audio_attn_weights_enc, video_enc_inter_outputs_class, video_enc_inter_outputs_count, video_enc_inter_outputs_coords, audio_enc_inter_outputs_class, audio_enc_inter_outputs_count, audio_enc_inter_outputs_coords

    def prepare_decoder_input_query(self, batch_size, query_embed):
        '''
        param: batch_size
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
        output, reference_points, video_sampling_locations_dec, video_attn_weights_dec, audio_sampling_locations_dec, audio_attn_weights_dec = self.decoder(*kargs)
        return output, reference_points, video_sampling_locations_dec, video_attn_weights_dec, audio_sampling_locations_dec, audio_attn_weights_dec



class MultimodalSparseDeformableTransformerEncoderLayer(nn.Module):
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

        self.isSparse = True

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


    def forward(self, video_src, audio_src, video_input,  audio_input, video_tgt=None, audio_tgt=None):
        '''
        param: src (batch_size, sum of num_token in all level, d_model)
        param: pos (batch_size, sum of num_token in all level, d_model) #  lvl_pos_embed_flatten
        param: reference_points (batch_size, sum of num_token in all level, num_feature_levels, 1)
        param: temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: padding_mask (batch_size, sum of num_tokens in all level)
        param: video_tgt
        param: audio_tgt
        
        return: audio_attended_visual: (batch_size, sum of num_token in all level, d_model) 
        return: visual_attended_audio: (batch_size, sum of num_token in all level, d_model) 
        return: video_sampling_locations 
        return: video_attn_weights 
        return: audio_sampling_locations 
        return: audio_attn_weights
        '''

        if video_tgt is None or audio_tgt is None:
            #   self attention for video
            video_src2, _, _ = self.self_attn(self.with_pos_embed(video_src, video_input['lvl_pos_embed_flatten']), video_input['reference_points'], 
                                                video_src, video_input['temporal_shapes'], video_input['level_start_index'], video_input['mask_flatten'], self.isSparse)
            video_src = video_src + self.dropout1(video_src2)
            video_src = self.norm1(video_src)

            #   self attention for audio
            audio_src2,  _, _ = self.self_attn(self.with_pos_embed(audio_src, audio_input['lvl_pos_embed_flatten']), audio_input['reference_points'], 
                                                audio_src, audio_input['temporal_shapes'], audio_input['level_start_index'], audio_input['mask_flatten'], self.isSparse)
            audio_src = audio_src + self.dropout1(audio_src2)
            audio_src = self.norm1(audio_src)

            #   multimodal attention
            visual_attended_audio, audio_sampling_locations, audio_attn_weights = self.self_attn(audio_src, audio_input['reference_points'], video_src, 
                                                                                                video_input['temporal_shapes'], video_input['level_start_index'], video_input['mask_flatten'], self.isSparse)
            audio_attended_visual, video_sampling_locations, video_attn_weights = self.self_attn(video_src, video_input['reference_points'], audio_src, 
                                                                                                audio_input['temporal_shapes'], audio_input['level_start_index'], audio_input['mask_flatten'], self.isSparse)

            # ffn
            visual_attended_audio = self.forward_ffn(visual_attended_audio)
            audio_attended_visual = self.forward_ffn(audio_attended_visual)

            return audio_attended_visual, visual_attended_audio, video_sampling_locations, video_attn_weights, audio_sampling_locations, audio_attn_weights
        else:
            #   self attention for video
            video_tgt2, _, _ = self.self_attn(self.with_pos_embed(video_tgt, video_input['lvl_pos_embed_flatten']), video_input['reference_points'], 
                                                video_src, video_input['temporal_shapes'], video_input['level_start_index'], video_input['mask_flatten'], self.isSparse)
            video_tgt = video_tgt + self.dropout1(video_tgt2)
            video_tgt = self.norm1(video_tgt)

            #   self attention for audio
            audio_tgt2,  _, _ = self.self_attn(self.with_pos_embed(audio_tgt, audio_input['lvl_pos_embed_flatten']), audio_input['reference_points'], 
                                                audio_src, audio_input['temporal_shapes'], audio_input['level_start_index'], audio_input['mask_flatten'], self.isSparse)
            audio_tgt = audio_tgt + self.dropout1(audio_tgt2)
            audio_tgt = self.norm1(audio_tgt)

            #   multimodal attention
            visual_attended_audio, audio_sampling_locations, audio_attn_weights = self.self_attn(audio_tgt, audio_input['reference_points'], video_src, 
                                                                                                video_input['temporal_shapes'], video_input['level_start_index'], video_input['mask_flatten'], self.isSparse)
            audio_attended_visual, video_sampling_locations, video_attn_weights = self.self_attn(video_tgt, video_input['reference_points'], audio_src, 
                                                                                                audio_input['temporal_shapes'], audio_input['level_start_index'], audio_input['mask_flatten'], self.isSparse)

            # ffn
            visual_attended_audio = self.forward_ffn(visual_attended_audio)
            audio_attended_visual = self.forward_ffn(audio_attended_visual)

            return audio_attended_visual, visual_attended_audio, video_sampling_locations, video_attn_weights, audio_sampling_locations, audio_attn_weights



class MultimodalSparseDeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, d_model):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
         # hack implementation
        self.aux_heads = False
        self.video_class_embedding = None
        self.video_count_head = None
        self.video_segment_embedding = None
        self.audio_class_embedding = None
        self.audio_count_head = None
        self.audio_segment_embedding = None

    @staticmethod
    def get_reference_points(temporal_shapes, valid_ratios, device):
        '''
        :param temporal_shapes (num_feature_levels)    #   list of num token at each level [1500,  750,  375,  188]
        :param valid_ratios (batch_size, num_feature_levels)
        :param device - string

        :return reference_points (batch_size, sum of num_token in all level, num_feature_levels, 1)
        '''
        reference_points_list = []
        for lvl, (L_) in enumerate(temporal_shapes):
            ref = torch.linspace(0.5, L_ - 0.5, L_, dtype=torch.float32, device=device)    # Creates a one-dimensional tensor of size 3rd param whose values are evenly spaced from 1st param to 2nd param
            ref = ref.reshape(-1)[None] / (valid_ratios[:, None, lvl] * L_)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        reference_points = reference_points[:,:,:,None]
        return reference_points

    def forward(self, video_input, audio_input):
        """
        param: src (batch_size, sum of num_token in all level, d_model)
        param: temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: valid_ratios (batch_size, num_feature_levels)
        param: pos (batch_size, sum of num_token in all level, d_model) #  lvl_pos_embed_flatten
        param: padding_mask (batch_size, sum of num_tokens in all level)
    

        return: memory -> (audio_attended_visual, visual_attended_audio)
        audio_attended_visual: (batch_size, sum of num_token in all level, d_model) 
        visual_attended_audio: (batch_size, sum of num_token in all level, d_model) 


        """
        
        # print("output_proposals", output_proposals.shape)
        if self.aux_heads:
            assert video_input['backbone_output_proposals'] is not None and audio_input['backbone_output_proposals'] is not None
        else:
            assert video_input['backbone_output_proposals'] is None and audio_input['backbone_output_proposals'] is None

        video_output = video_input['src_flatten']
        video_sparsified_keys = False if video_input['backbone_topk_proposals'] is None else True
        video_input['reference_points'] = self.get_reference_points(video_input['temporal_shapes'], video_input['valid_ratios'], device=video_input['src_flatten'].device)  # (batch_size, sum of num_token in all level, num_feature_levels, 1)
        video_reference_points_orig = video_input['reference_points']
        video_pos_orig = video_input['lvl_pos_embed_flatten']
        video_output_proposals_orig = video_input['backbone_output_proposals']
        video_sampling_locations_enc = []
        video_attn_weights_enc = []

        audio_output = audio_input['src_flatten']
        audio_sparsified_keys = False if audio_input['backbone_topk_proposals'] is None else True
        audio_input['reference_points'] = self.get_reference_points(audio_input['temporal_shapes'], audio_input['valid_ratios'], device=audio_input['src_flatten'].device)  # (batch_size, sum of num_token in all level, num_feature_levels, 1)
        audio_reference_points_orig = audio_input['reference_points']
        audio_pos_orig = audio_input['lvl_pos_embed_flatten']
        audio_output_proposals_orig = audio_input['backbone_output_proposals']
        audio_sampling_locations_enc = []
        audio_attn_weights_enc = []
        if self.aux_heads:
            video_enc_inter_outputs_class = []
            video_enc_inter_outputs_count = []
            video_enc_inter_outputs_coords = []
            audio_enc_inter_outputs_class = []
            audio_enc_inter_outputs_count = []
            audio_enc_inter_outputs_coords = []
      
        if video_sparsified_keys:
            assert video_input['backbone_topk_proposals'] is not None
            B_, N_, S_, P_ = video_input['reference_points'].shape
            video_input['reference_points'] = torch.gather(video_input['reference_points'].view(B_, N_, -1), 1, video_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, S_*P_)).view(B_, -1, S_, P_)
            video_tgt = torch.gather(video_output, 1, video_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, video_output.size(-1)))
            video_input['lvl_pos_embed_flatten'] = torch.gather(video_input['lvl_pos_embed_flatten'], 1, video_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, video_input['lvl_pos_embed_flatten'].size(-1)))
            if video_input['backbone_output_proposals'] is not None:
                video_input['backbone_output_proposals'] = video_input['backbone_output_proposals'].gather(1, video_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, video_input['backbone_output_proposals'].size(-1)))
        else:
            video_tgt = None

        if audio_sparsified_keys:
            assert audio_input['backbone_topk_proposals'] is not None
            B_, N_, S_, P_ = audio_input['reference_points'].shape
            audio_input['reference_points'] = torch.gather(audio_input['reference_points'].view(B_, N_, -1), 1, audio_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, S_*P_)).view(B_, -1, S_, P_)
            audio_tgt = torch.gather(audio_output, 1, audio_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, audio_output.size(-1)))
            audio_input['lvl_pos_embed_flatten'] = torch.gather(audio_input['lvl_pos_embed_flatten'], 1, audio_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, audio_input['lvl_pos_embed_flatten'].size(-1)))
            if audio_input['backbone_output_proposals'] is not None:
                audio_input['backbone_output_proposals'] = audio_input['backbone_output_proposals'].gather(1, audio_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, audio_input['backbone_output_proposals'].size(-1)))
        else:
            audio_tgt = None



        for lid, layer in enumerate(self.layers):
            # if tgt is None: self-attention / if tgt is not None: cross-attention w.r.t. the target queries
            audio_attended_visual, visual_attended_audio, video_sampling_locations, video_attn_weights, audio_sampling_locations, audio_attn_weights = layer(video_output, audio_output, video_input, audio_input, 
                                                                                                                                                                video_tgt=video_tgt if video_sparsified_keys else None, audio_tgt=audio_tgt if audio_sparsified_keys else None)
            video_sampling_locations_enc.append(video_sampling_locations)
            video_attn_weights_enc.append(video_attn_weights)
            audio_sampling_locations_enc.append(audio_sampling_locations)
            audio_attn_weights_enc.append(audio_attn_weights)

            if video_sparsified_keys:                
                if video_input['sparse_token_nums'] is None:
                    video_output = video_output.scatter(1, video_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, audio_attended_visual.size(-1)), audio_attended_visual)
                else:
                    video_outputs = []
                    for i in range(video_input['backbone_topk_proposals'].shape[0]):
                        video_outputs.append(video_output[i].scatter(0, video_input['backbone_topk_proposals'][i][:video_input['sparse_token_nums'][i]].unsqueeze(-1).repeat(1, audio_attended_visual.size(-1)), audio_attended_visual[i][:video_input['sparse_token_nums'][i]]))
                    video_output = torch.stack(video_outputs)
            else:
                video_output = audio_attended_visual

            if audio_sparsified_keys:                
                if audio_input['sparse_token_nums'] is None:
                    audio_output = audio_output.scatter(1, audio_input['backbone_topk_proposals'].unsqueeze(-1).repeat(1, 1, visual_attended_audio.size(-1)), visual_attended_audio)
                else:
                    audio_outputs = []
                    for i in range(audio_input['backbone_topk_proposals'].shape[0]):
                        audio_outputs.append(video_output[i].scatter(0, audio_input['backbone_topk_proposals'][i][:audio_input['sparse_token_nums'][i]].unsqueeze(-1).repeat(1, visual_attended_audio.size(-1)), visual_attended_audio[i][:audio_input['sparse_token_nums'][i]]))
                    audio_output = torch.stack(audio_outputs)
            else:
                audio_output = audio_attended_visual

            if self.aux_heads and lid < self.num_layers - 1:
                # feed outputs to aux. heads
                video_output_class = self.video_class_embedding[lid](audio_attended_visual)
                video_output_count = predict_event_num(self.video_count_head[lid], audio_attended_visual)
                video_output_offset = self.video_segment_embedding[lid](audio_attended_visual)
                video_output_coords_unact = video_input['backbone_output_proposals'] + video_output_offset

                audio_output_class = self.audio_class_embedding[lid](visual_attended_audio)
                audio_output_count = predict_event_num(self.audio_count_head[lid], visual_attended_audio)
                audio_output_offset = self.audio_segment_embedding[lid](visual_attended_audio)
                audio_output_coords_unact = audio_input['backbone_output_proposals'] + audio_output_offset

                # values to be used for loss compuation
                video_enc_inter_outputs_class.append(video_output_class)
                video_enc_inter_outputs_count.append(video_output_count)
                video_enc_inter_outputs_coords.append(video_output_coords_unact.sigmoid())

                 # values to be used for loss compuation
                audio_enc_inter_outputs_class.append(audio_output_class)
                audio_enc_inter_outputs_count.append(audio_output_count)
                audio_enc_inter_outputs_coords.append(audio_output_coords_unact.sigmoid())

        # Change dimension from [num_layer, batch_size, ...] to [batch_size, num_layer, ...]
        video_sampling_locations_enc = torch.stack(video_sampling_locations_enc, dim=1)
        video_attn_weights_enc = torch.stack(video_attn_weights_enc, dim=1)

        # Change dimension from [num_layer, batch_size, ...] to [batch_size, num_layer, ...]
        audio_sampling_locations_enc = torch.stack(audio_sampling_locations_enc, dim=1)
        audio_attn_weights_enc = torch.stack(audio_attn_weights_enc, dim=1)
        

        if self.aux_heads:
            return video_output, video_sampling_locations_enc, video_attn_weights_enc, audio_output, audio_sampling_locations_enc, audio_attn_weights_enc, video_enc_inter_outputs_class, video_enc_inter_outputs_count, video_enc_inter_outputs_coords, audio_enc_inter_outputs_class, audio_enc_inter_outputs_count, audio_enc_inter_outputs_coords 
        else:
            return video_output, video_sampling_locations_enc, video_attn_weights_enc, audio_output, audio_sampling_locations_enc, audio_attn_weights_enc, None, None, None, None



class MultimodalSparseDeformableTransformerDecoderLayer(nn.Module):
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

        self.isSparse = True
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

    # TODO - check key_padding_mask (~mask??)
    def forward(self, tgt, query_pos, reference_points_input_video, reference_points_input_audio, query_mask, video_src, video_temporal_shapes, video_level_start_index,
                video_src_padding_mask, audio_src, audio_temporal_shapes, audio_level_start_index,
                audio_src_padding_mask):
        """
        param: tgt (batch_size, num_queries, d_model)
        param: query_pos (num_queries, d_model * 2)
        param: reference_points (batch_size, num_queries, 1)
        param: src (batch_size, sum of num_token in all level, d_model)
        param: src_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor    
        param: src_padding_mask (batch_size, sum of num_tokens in all level)
        param: query_mask (batch_size, num_queries)

        return: output (batch_size, num_queries, d_model)
        """
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=~query_mask)[
            0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)


        # cross attention for video
        tgt_video, video_sampling_locations, video_attn_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points_input_video,
                               video_src, video_temporal_shapes, video_level_start_index, video_src_padding_mask, self.isSparse)
        tgt_video = tgt + self.dropout1(tgt_video)
        tgt_video = self.norm1(tgt_video)


        # cross attention for audio
        tgt_audio, audio_sampling_locations, audio_attn_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points_input_audio,
                               audio_src, audio_temporal_shapes, audio_level_start_index, audio_src_padding_mask, self.isSparse)
        tgt_audio = tgt + self.dropout1(tgt_audio)
        tgt_audio = self.norm1(tgt_audio)

        # bridge
        tgt = torch.cat([tgt_video, tgt_audio], dim=-1) #   (batch_size, num_queries, 2*dmodel)
        tgt = self.norm4(tgt)
        tgt = self.linear3(tgt) #   (batch_size, num_queries, dmodel)
        tgt = self.dropout5(tgt)
        tgt = self.activation(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt, tgt_video, tgt_audio, video_sampling_locations, video_attn_weights, audio_sampling_locations, audio_attn_weights



class MultimodalSparseDeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_head = None
        # self.bbox_embed = None
        # self.class_embed = None
        # self.count_head = None

    def forward(self, tgt, reference_points, video_src, video_input, audio_src, audio_input, query_pos=None, query_padding_mask=None, disable_iterative_refine=False):
        """
        param: tgt (batch_size, num_queries, d_model)
        param: reference_points (batch_size, num_queries, 1)
        param: src (batch_size, sum of num_token in all level, d_model)
        param: src_temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: src_level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: src_valid_ratios (batch_size, num_feature_levels)
        param: query_pos (num_queries, d_model * 2)
        param: src_padding_mask (batch_size, sum of num_tokens in all level)
        param: query_padding_mask (batch_size, num_queries)
        param: disable_iterative_refine bool
        
        return: output: (number of decoder_layers, batch_size, num_queries, d_model)
        return: reference_points: (number of decoder_layers, batch_size, num_queries, 1)
        """
        output = tgt    #   (batch_size, num_queries, d_model)

        intermediate = []
        intermediate_reference_points = []
        video_sampling_locations_dec = []
        video_attn_weights_dec = []
        audio_sampling_locations_dec = []
        audio_attn_weights_dec = []
        bs = tgt.shape[0]
        # print("reference_points", reference_points.shape)
        # print(video_input['valid_ratios'].shape, "Fffff")
        for lid, layer in enumerate(self.layers):
            # print("reference_points", reference_points.shape)
            # print(video_input['valid_ratios'], "Fffff")
            if reference_points.shape[-1] == 2:
                reference_points_input_video = reference_points[:, :, None] \
                                         * torch.stack([video_input['valid_ratios'], video_input['valid_ratios']], -1)[:, None]
                reference_points_input_audio = reference_points[:, :, None] \
                                         * torch.stack([audio_input['valid_ratios'], audio_input['valid_ratios']], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 1
                reference_points_input_video = reference_points[:, :, None] * video_input['valid_ratios'][:, None, :, None]  #   (batch_size, num_queries, num_feature_levels, 1)
                reference_points_input_audio = reference_points[:, :, None] * audio_input['valid_ratios'][:, None, :, None]

            output, tgt_video, tgt_audio, video_sampling_locations, video_attn_weights, audio_sampling_locations, audio_attn_weights = layer(output, query_pos, reference_points_input_video, reference_points_input_audio, query_padding_mask, video_src, video_input['temporal_shapes'], video_input['level_start_index'], video_input['mask_flatten'], audio_src, audio_input['temporal_shapes'], audio_input['level_start_index'], audio_input['mask_flatten'])    #   (batch_size, num_queries, d_model)
            
            video_sampling_locations_dec.append(video_sampling_locations)
            video_attn_weights_dec.append(video_attn_weights)

            audio_sampling_locations_dec.append(audio_sampling_locations)
            audio_attn_weights_dec.append(audio_attn_weights)

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

        video_sampling_locations_dec = torch.stack(video_sampling_locations_dec, dim=1)
        video_attn_weights_dec = torch.stack(video_attn_weights_dec, dim=1)
        audio_sampling_locations_dec = torch.stack(audio_sampling_locations_dec, dim=1)
        audio_attn_weights_dec = torch.stack(audio_attn_weights_dec, dim=1)

        if self.return_intermediate:
            intermediate_outputs = torch.stack(intermediate)
            intermediate_reference_points = torch.stack(intermediate_reference_points)
            return intermediate_outputs, intermediate_reference_points, video_sampling_locations_dec, video_attn_weights_dec, audio_sampling_locations_dec, audio_attn_weights_dec

        return output, reference_points, video_sampling_locations_dec, video_attn_weights_dec, audio_sampling_locations_dec, audio_attn_weights_dec
    

class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )
    
    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


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


def build_multimodal_sparse_deforamble_transformer(args):
    return MultimodalSparseDeformableTransformer(
        d_model=args.d_model,
        num_head=args.num_heads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.transformer_ff_dim,
        dropout=args.transformer_dropout_prob,
        activation="relu",
        return_intermediate_dec=args.return_intermediate,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        rho=args.rho,
        use_enc_aux_loss=args.use_enc_aux_loss,
        eff_query_init=args.eff_query_init,
        eff_specific_head=args.eff_specific_head
        )