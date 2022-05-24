import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_
from ..modules.misc_modules import inverse_sigmoid, predict_event_num 
from ..modules.attention import MSDeformAttn

class SparseDeformableTransformer(nn.Module):
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

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, num_head, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, self.d_model)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, num_head, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

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
            wh = torch.full(grid.shape, 0.05 * (2.0 ** lvl), device=grid.device)    #   effiecint code for above line (batch_size, num_token in L layer)
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


        return src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten,  backbone_output_proposals, backbone_topk_proposals, backbone_mask_prediction, sparse_token_nums   


    def forward_encoder(self, src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                        mask_flatten, backbone_output_proposals, backbone_topk_proposals, sparse_token_nums):
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
        output_proposals = backbone_output_proposals if self.use_enc_aux_loss else None 
        output, sampling_locations_enc, attn_weights_enc, enc_inter_outputs_class, enc_inter_outputs_count, enc_inter_outputs_coords = self.encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                                mask_flatten, backbone_topk_proposals, output_proposals, sparse_token_nums)

        return output, sampling_locations_enc, attn_weights_enc, enc_inter_outputs_class, enc_inter_outputs_count, enc_inter_outputs_coords

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
        hs, inter_references_out,  sampling_locations_dec, attn_weights_dec = self.decoder(*kargs)
        return hs, inter_references_out,  sampling_locations_dec, attn_weights_dec



class DeformableTransformerEncoderLayer(nn.Module):
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

    def forward(self, src, pos, reference_points, temporal_shapes, level_start_index, padding_mask=None, tgt=None):
        '''
        param: src (batch_size, sum of num_token in all level, d_model)
        param: pos (batch_size, sum of num_token in all level, d_model) #  lvl_pos_embed_flatten
        param: reference_points (batch_size, sum of num_token in all level, num_feature_levels, 1)
        param: temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: padding_mask (batch_size, sum of num_tokens in all level)
        
        return: output: (batch_size, sum of num_token in all level, d_model) 
        '''

        if tgt is None:
            # self attention
            src2, sampling_locations, attn_weights = self.self_attn(self.with_pos_embed(src, pos),
                                reference_points, src, temporal_shapes,
                                level_start_index, padding_mask, is_sparse=True)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            # torch.Size([2, 13101, 256])

            # ffn
            src = self.forward_ffn(src)

            return src, sampling_locations, attn_weights
        else:
            tgt2, sampling_locations, attn_weights = self.self_attn(self.with_pos_embed(tgt, pos),
                                reference_points, src, temporal_shapes,
                                level_start_index, padding_mask, is_sparse=True)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # ffn
            tgt = self.forward_ffn(tgt)

            return tgt, sampling_locations, attn_weights
            


class DeformableTransformerEncoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers, d_model):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
         # hack implementation
        self.aux_heads = False
        self.class_embedding = None
        self.count_head = None
        self.segment_embedding = None

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

    def forward(self, src, temporal_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, backbone_topk_proposals=None, output_proposals=None, sparse_token_nums=None):
        """
        param: src (batch_size, sum of num_token in all level, d_model)
        param: temporal_shapes (num_feature_levels)    #   list of num token at each level
        param: level_start_index (num_feature_levels)  #   list to find the start index of each level from flatten tensor
        param: valid_ratios (batch_size, num_feature_levels)
        param: pos (batch_size, sum of num_token in all level, d_model) #  lvl_pos_embed_flatten
        param: padding_mask (batch_size, sum of num_tokens in all level)
        
        return: output: (batch_size, sum of num_token in all level, d_model) #   Multi-scale frame features

        """
        # print("output_proposals", output_proposals.shape)
        if self.aux_heads:
            assert output_proposals is not None
        else:
            assert output_proposals is None

        output = src
        sparsified_keys = False if backbone_topk_proposals is None else True
        reference_points = self.get_reference_points(temporal_shapes, valid_ratios, device=src.device)  # (batch_size, sum of num_token in all level, num_feature_levels, 1)
        reference_points_orig = reference_points
        pos_orig = pos
        output_proposals_orig = output_proposals
        sampling_locations_enc = []
        attn_weights_enc = []
        if self.aux_heads:
            enc_inter_outputs_class = []
            enc_inter_outputs_count = []
            enc_inter_outputs_coords = []
      
      
        if sparsified_keys:
            assert backbone_topk_proposals is not None
            B_, N_, S_, P_ = reference_points.shape
            reference_points = torch.gather(reference_points.view(B_, N_, -1), 1, backbone_topk_proposals.unsqueeze(-1).repeat(1, 1, S_*P_)).view(B_, -1, S_, P_)
            tgt = torch.gather(output, 1, backbone_topk_proposals.unsqueeze(-1).repeat(1, 1, output.size(-1)))
            pos = torch.gather(pos, 1, backbone_topk_proposals.unsqueeze(-1).repeat(1, 1, pos.size(-1)))
            if output_proposals is not None:
                output_proposals = output_proposals.gather(1, backbone_topk_proposals.unsqueeze(-1).repeat(1, 1, output_proposals.size(-1)))
        else:
            tgt = None
        for lid, layer in enumerate(self.layers):
            # if tgt is None: self-attention / if tgt is not None: cross-attention w.r.t. the target queries
            tgt, sampling_locations, attn_weights = layer(output, pos, reference_points, temporal_shapes, level_start_index, padding_mask, 
                        tgt=tgt if sparsified_keys else None)
            sampling_locations_enc.append(sampling_locations)
            attn_weights_enc.append(attn_weights)
            if sparsified_keys:                
                if sparse_token_nums is None:
                    output = output.scatter(1, backbone_topk_proposals.unsqueeze(-1).repeat(1, 1, tgt.size(-1)), tgt)
                else:
                    outputs = []
                    for i in range(backbone_topk_proposals.shape[0]):
                        outputs.append(output[i].scatter(0, backbone_topk_proposals[i][:sparse_token_nums[i]].unsqueeze(-1).repeat(1, tgt.size(-1)), tgt[i][:sparse_token_nums[i]]))
                    output = torch.stack(outputs)
            else:
                output = tgt
            
            if self.aux_heads and lid < self.num_layers - 1:
                # feed outputs to aux. heads
                output_class = self.class_embedding[lid](tgt)
                output_count = predict_event_num(self.count_head[lid], tgt)
                output_offset = self.segment_embedding[lid](tgt)
                output_coords_unact = output_proposals + output_offset
                # values to be used for loss compuation
                enc_inter_outputs_class.append(output_class)
                enc_inter_outputs_count.append(output_count)
                enc_inter_outputs_coords.append(output_coords_unact.sigmoid())

        # Change dimension from [num_layer, batch_size, ...] to [batch_size, num_layer, ...]
        sampling_locations_enc = torch.stack(sampling_locations_enc, dim=1)
        attn_weights_enc = torch.stack(attn_weights_enc, dim=1)

        if self.aux_heads:
            return output, sampling_locations_enc, attn_weights_enc, enc_inter_outputs_class, enc_inter_outputs_count, enc_inter_outputs_coords
        else:
            return output, sampling_locations_enc, attn_weights_enc, None, None, None
        


class DeformableTransformerDecoderLayer(nn.Module):
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

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    # TODO - check key_padding_mask (~mask??)
    def forward(self, tgt, query_pos, reference_points, src, src_temporal_shapes, level_start_index,
                src_padding_mask=None, query_mask=None):

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

        # cross attention
        tgt2, sampling_locations, attn_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_temporal_shapes, level_start_index, src_padding_mask, is_sparse=True)


        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt, sampling_locations, attn_weights


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_head = None
        # self.segment_embedding = None
        # self.class_embedding = None
        # self.count_head= None

    def forward(self, tgt, reference_points, src, src_temporal_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, query_padding_mask=None, disable_iterative_refine=False):
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
        sampling_locations_enc = []
        attn_weights_enc = []
        bs = tgt.shape[0]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.stack([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 1
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None, :, None]  #   (batch_size, num_queries, num_feature_levels, 1)
            output, sampling_locations, attn_weights = layer(output, query_pos, reference_points_input, src, src_temporal_shapes, src_level_start_index, src_padding_mask, query_padding_mask)    #   (batch_size, num_queries, d_model)
            
            sampling_locations_enc.append(sampling_locations)
            attn_weights_enc.append(attn_weights)

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

        sampling_locations_enc = torch.stack(sampling_locations_enc, dim=1)
        attn_weights_enc = torch.stack(attn_weights_enc, dim=1)

        if self.return_intermediate:
            intermediate_outputs = torch.stack(intermediate)
            intermediate_reference_points = torch.stack(intermediate_reference_points)
            return intermediate_outputs, intermediate_reference_points, sampling_locations_enc, attn_weights_enc

        return output, reference_points, sampling_locations_enc, attn_weights_enc
    

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


def build_sparse_deformable_transformer(args):
    return SparseDeformableTransformer(
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