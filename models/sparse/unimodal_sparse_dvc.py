""" 
DVC model for event segmentation and captioning
"""

import math
from math import floor, ceil
import copy

from pprint import pprint

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .unimodal_sparse_deformable_transformer import build_sparse_deforamble_transformer
from ..base_encoder import build_base_encoder
from ..unimodal_caption_decoder import build_unimodal_caption_decoder

from ..modules.embedding_layers import PositionEmbeddingVideoSine
from ..modules.layers import FFN, ContextMaskModel
from ..modules.misc_modules import decide_two_stage, inverse_sigmoid, predict_event_num_with_depth

from ..load_weights import load_positional_embeddings

from utils.preds_postprocess import get_src_permutation_idx, denormalize_segments



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# TODO - check devices for tensors
# TODOD - is_sparse flag in init
class UnimodalSparseDVC(nn.Module):
    def __init__(self, input_modalities, num_queries, d_model, num_classes, aux_loss, max_eseq_length, sparse_detr_args):
        
        """
        UnimodalSparseDVC model
        """

        super(UnimodalSparseDVC, self).__init__()
        
        self.input_modalities = input_modalities
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.num_classes = num_classes

        self.query_embedding = nn.Embedding(num_queries, d_model * 2)

        self.class_embedding_encoder = nn.Linear(d_model, num_classes + 1)
        self.class_embedding_decoder = nn.Linear(d_model, num_classes + 1)

        self.segment_embedding_encoder = FFN(in_dim=d_model, hidden_dim=d_model, out_dim=2, num_layers=3)
        self.segment_embedding_decoder = FFN(in_dim=d_model, hidden_dim=d_model, out_dim=2, num_layers=3)
        
        self.count_head_encoder = nn.Linear(d_model, max_eseq_length + 1)
        self.count_head_decoder = nn.Linear(d_model, max_eseq_length + 1)

        assert 'video' in input_modalities or 'audio' in input_modalities, f'input_modalities should contain one of "video" or "audio". You have {input_modalities}'

        self.pos_embed = PositionEmbeddingVideoSine(d_model//2, normalize=True)

        self.rho = sparse_detr_args.rho
        self.use_enc_aux_loss = sparse_detr_args.use_enc_aux_loss

        self.base_encoder = build_base_encoder(sparse_detr_args)

        # TODO - do all this in init_weights()
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embedding_encoder.bias.data = torch.ones(num_classes + 1) * bias_value
        self.class_embedding_decoder.bias.data = torch.ones(num_classes + 1) * bias_value

        nn.init.constant_(self.segment_embedding_encoder.layers[-1].weight.data, 0.)
        nn.init.constant_(self.segment_embedding_encoder.layers[-1].bias.data, 0.)
        nn.init.constant_(self.segment_embedding_decoder.layers[-1].weight.data, 0.)
        nn.init.constant_(self.segment_embedding_decoder.layers[-1].bias.data[:2], 0.)
        nn.init.constant_(self.segment_embedding_decoder.layers[-1].bias.data[2:], -2.0)


        # Unimodal Sparse DETR
        self.unimodal_sparse_transformer = build_sparse_deforamble_transformer(sparse_detr_args)
        
        if sparse_detr_args.use_enc_aux_loss:
            self.unimodal_sparse_transformer.encoder.aux_heads = True
            self.unimodal_sparse_transformer.encoder.class_embedding = self.class_embedding_encoder
            self.unimodal_sparse_transformer.encoder.count_head = self.count_head_encoder
            self.unimodal_sparse_transformer.encoder.segment_embedding = self.segment_embedding_encoder

        
        # self.init_weights()


    # TODO - use log softmax? 
    # TODO - pos embed (static, learned)
    # TODO - check pos embed for all layers
    def forward(self, obj):

        """
        Performs a forward pass on the UnimodalSparseDVC model which consists of the encoders, proposal decoder and caption decoder
  
        Parameters:
            obj (collections.defaultdict): Consisitng of various keys including 
                                           video_tensor (batch_size, in_channels, num_frames, img_size, img_size)
                                           video_mask (batch_size, num_frames)
                                           video_length (batch_size, 3) - num_frames, duration, gt_target_segments
        
        Returns:
            out (dictionary) : It returns a dict with the following elements:
                                - "pred_logits": the classification logits (including no-object) for all queries
                                                    shape (batch_size, num_queries, num_classes + 1)
                                - "pred_segments": The normalized segments for all queries, represented as
                                                (center_offset, length). Shape (batch_size, num_queries, 2)
            ???????
            values are normalized in [0, 1]
            relative to the size of each individual image (disregarding possible padding).
            See PostProcess for information on how to retrieve the unnormalized bounding box.

                                - "pred_captions": All captions in a batch with shape (total_caption_num, max_caption_length - 1, vocab_size)

                                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                                    dictionaries containing the two above keys for each decoder layer.

            indices (list): matching between the outputs of the last layer and the targets
                            list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))

        """

        video = obj['video_tensor']    # (batch_size, num_tokens_v, d_model)
        video_mask = obj['video_mask']    # (batch_size, num_tokens_v)
        
        durations = obj['video_length'][:, 1]   # (batch_size)

        # audio = obj['audio_tensor']    # (batch_size, num_tokens_a, d_model)
        # audio_mask = obj['audio_mask']    # (batch_size, num_tokens_a)
        
        batch_size, _, _ = video.shape

        # Base Encoder - for multi-scale features
        if 'video' in self.input_modalities: 
            srcs, masks, pos = self.base_encoder(video, video_mask, durations, self.pos_embed)
        # else:
        #     srcs, masks, pos = self.base_encoder(audio, audio_mask, durations, self.pos_embed)

        # Forword Encoder
        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, backbone_output_proposals, backbone_topk_proposals, backbone_mask_prediction, sparse_token_nums = self.unimodal_sparse_transformer.prepare_encoder_inputs(srcs, masks, pos)

        memory, sampling_locations_enc, attn_weights_enc, enc_inter_outputs_class, enc_inter_outputs_count, enc_inter_outputs_segments = self.unimodal_sparse_transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, backbone_output_proposals, backbone_topk_proposals, sparse_token_nums)    

        # Forword Decoder
        # TODO - check proposals_mask (key_padding_mask in deformable_transformer (~mask??))
        # TODO - see transformer_input_type = "gt_proposals"
        transformer_input_type = "queries"
        gt_boxes = None
        gt_boxes_mask = None
        criterion = None
        two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type, gt_boxes, gt_boxes_mask, criterion)

        if two_stage:
            init_reference, tgt, reference_points, query_embed = self.unimodal_sparse_transformer.prepare_decoder_input_proposal(proposals)
        else:
            query_embedding_weight = self.query_embedding.weight
            proposals_mask = torch.ones(batch_size, query_embedding_weight.shape[0], device=query_embedding_weight.device).bool()  #   (batch_size, num_queries)
            init_reference, tgt, reference_points, query_embedding_weight = self.unimodal_sparse_transformer.prepare_decoder_input_query(batch_size, query_embedding_weight)


        query_features, inter_references, sampling_locations_dec, attn_weights_dec = self.unimodal_sparse_transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                                                                                    level_start_index, valid_ratios,  query_embedding_weight, 
                                                                                                                                    mask_flatten, proposals_mask, disable_iterative_refine)
        
        # (1, batch_size, num_queries, num_classes + 1) OR (depth, batch_size, num_queries, num_classes + 1)
        outputs_class = self.class_embedding_decoder(query_features).softmax(dim=-1)

        # (1, batch_size, num_queries, 2) OR (depth, batch_size, num_queries, 2)
        outputs_segment = self.segment_embedding_decoder(query_features)

        # (1, batch_size, max_eseq_length + 1) OR (depth, batch_size, max_eseq_length + 1)
        outputs_count = predict_event_num_with_depth(self.count_head_decoder, query_features)

        assert init_reference is not None and inter_references is not None
        reference = inter_references
        reference[0] = init_reference
        reference[1:] = inter_references[:-1].clone()    # TODO - clone() - error if removed
        
        reference = inverse_sigmoid(reference)
        if reference.shape[-1] == 2:
            outputs_segment += reference
        else:
            assert reference.shape[-1] == 1
            outputs_segment[..., :2] += reference
        
        outputs_segment = outputs_segment.sigmoid()

        out = {'pred_logits': outputs_class[-1], 
                'pred_segments': outputs_segment[-1],
                'pred_count': outputs_count[-1],
                'sampling_locations_enc': sampling_locations_enc,
                'attn_weights_enc': attn_weights_enc,
                'sampling_locations_dec': sampling_locations_dec,
                'attn_weights_dec': attn_weights_dec,
                'temporal_shapes': temporal_shapes,
                'level_start_index': level_start_index
            }

        if backbone_topk_proposals is not None:
            out["backbone_topk_proposals"] = backbone_topk_proposals

        if self.rho:
            out["backbone_mask_prediction"] = backbone_mask_prediction

        if self.use_enc_aux_loss:
            out['aux_outputs_enc'] = self._set_aux_loss(enc_inter_outputs_class, enc_inter_outputs_segments, enc_inter_outputs_count, is_enc_aux=True)
        
        if self.rho:
            out["sparse_token_nums"] = sparse_token_nums

        out['mask_flatten'] = torch.cat([m.flatten(1) for m in masks], 1)
            
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment, outputs_count)

        return out, torch.argmax(outputs_class[-1], dim=-1)
            

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segment, outputs_count, is_enc_aux=False):
        if is_enc_aux:
            return [{'pred_logits': a, 'pred_segments': b, 'pred_count': c}
                for a, b, c in zip(outputs_class, outputs_segment, outputs_count)]
        
        else:
            return [{'pred_logits': a, 'pred_segments': b, 'pred_count': c}
                for a, b, c in zip(outputs_class[:-1], outputs_segment[:-1], outputs_count[:-1])]


    def make_tgt_mask(self, target, device):
        """
        Generates a mask that is a combination of a lookahead mask and a padding mask
        
        Parameters:
            target (Tensor): Tensor of dimension (batch_size, seq_len)
            tgt_padding_mask (Tensor): Padding mask of dimension (batch_size, seq_len)
        
        Returns:
            tgt_mask (Tensor): Tensor of dimention (batch_size, 1, seq_len, seq_len)
        """

        batch_size, seq_len = target.shape

        look_ahead_mask = 1 - torch.tril(torch.ones((seq_len, seq_len), device=device))

        return look_ahead_mask.bool()    # (seq_len, seq_len)


    def make_memory_mask(self):
        """
        Generates the memory padding mask
        
        Parameters:
            memory (Tensor): Tensor of dimension (batch_size, seq_len)
        
        Returns:
            memory_mask (Tensor): Tensor of dimension (batch_size, 1, 1, num_tokens)
        """
        pass


    def make_padding_mask(self, target):
        """
        Generates a padding mask (False where target contains PAD_TOKEN, True elsewhere)
        
        Parameters:
            target (Tensor): Tensor of dimension (batch_size, seq_len)
        
        Returns:
            tgt_padding_mask (Tensor): Tensor of dimention (batch_size, seq_len)
        """

        tgt_padding_mask = (target == self.vocab['<pad>'])
        return tgt_padding_mask

    
    def init_weights(self):

        """
        Initialises the weights and biases of the modules in the UnimodalSparseDVC model.
        These parameters include positional embeddings.
        """

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embedding.bias.data = torch.ones(self.num_classes + 1) * bias_value
        nn.init.constant_(self.segment_embedding.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embedding.layers[-1].bias.data, 0)
