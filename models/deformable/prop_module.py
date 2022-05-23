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
class PropUnimodalSparseDVC(nn.Module):
    def __init__(self, input_modalities, num_queries, d_model, num_classes, aux_loss, threshold, max_eseq_length, 
                detr_args):
        
        """
        UnimodalSparseDVC model for proposal generation
        """

        super(PropUnimodalSparseDVC, self).__init__()
        
        self.input_modalities = input_modalities
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.threshold = threshold

        self.query_embedding = nn.Embedding(num_queries, d_model * 2)

        self.class_embedding = nn.Linear(d_model, num_classes + 1)
        self.segment_embedding = FFN(in_dim=d_model, hidden_dim=d_model, out_dim=2, num_layers=3)

        self.count_head = nn.Linear(d_model, max_eseq_length + 1)

        self.matcher = matcher

        assert 'video' in input_modalities or 'audio' in input_modalities, f'input_modalities should contain one of "video" or "audio". You have {input_modalities}'

        self.pos_embed = PositionEmbeddingVideoSine(d_model//2, normalize=True)

        self.base_encoder = build_base_encoder(detr_args)
        
        # TODO - do all this in init_weights()
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embedding.bias.data = torch.ones(num_classes + 1) * bias_value
        nn.init.constant_(self.segment_embedding.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embedding.layers[-1].bias.data, 0)

        # TODO - return intermediate=False deos not output depth dimesntion (dim 0)
        # Unimodal Deformable DETR
        self.unimodal_deformable_transformer = build_unimodal_deformable_transformer(detr_args)
        
        # shared heads
        # num_pred = detr_args.dec_layers
        # nn.init.constant_(self.segment_embedding.layers[-1].bias.data[2:], -2.0)    # TODO - data[2:]??
        # self.class_embedding = nn.ModuleList([self.class_embedding for _ in range(num_pred)])
        # self.count_head = nn.ModuleList([self.count_head for _ in range(num_pred)])
        # self.segment_embedding = nn.ModuleList([self.segment_embedding for _ in range(num_pred)])

        # Context Module
        self.num_feature_levels = detr_args.num_feature_levels
        self.video_rescale_len = detr_args.video_rescale_len
        self.num_tokens = ceil(((2**self.num_feature_levels - 1) / 2**(self.num_feature_levels - 1)) * self.video_rescale_len)
        

        # self.init_weights()


    # TODO - use log softmax?
    # TODO - padding and src_mask for vid features as input to caption decoder  
    # TODO - pos embed (static, learned)
    # TODO - check pos embed for all layers
    def forward(self, obj, is_training=True, faster_eval=False):

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
        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.unimodal_deformable_transformer.prepare_encoder_inputs(srcs, masks, pos)

        # (batch_size, sum of num_tokens in all levels, d_model) - Multi-scale frame features
        memory = self.unimodal_deformable_transformer.forward_encoder(src_flatten, temporal_shapes, 
                                                                    level_start_index, valid_ratios, 
                                                                    lvl_pos_embed_flatten, mask_flatten)    

        # Forword Decoder
        # TODO - check proposals_mask (key_padding_mask in deformable_transformer (~mask??))
        transformer_input_type = "queries"
        gt_boxes = None
        gt_boxes_mask = None
        criterion = None
        two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type, gt_boxes, gt_boxes_mask, criterion)

        if two_stage:
            init_reference, tgt, reference_points, query_embedding = self.unimodal_deformable_transformer.prepare_decoder_input_proposal(proposals)
        else:
            query_embedding_weight = self.query_embedding.weight
            proposals_mask = torch.ones(batch_size, query_embedding_weight.shape[0], device=query_embedding_weight.device).bool()  #   (batch_size, num_queries)
            init_reference, tgt, reference_points, query_embedding_weight = self.unimodal_deformable_transformer.prepare_decoder_input_query(batch_size, query_embedding_weight)

        # query_features (depth, batch_size, num_queries, d_model)
        # inter_reference = (depth, batch_size, num_queries, 1)
        query_features, inter_references = self.unimodal_deformable_transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                level_start_index, valid_ratios, query_embedding_weight,
                                                                mask_flatten, proposals_mask, disable_iterative_refine)

        # (1, batch_size, num_queries, num_classes + 1) OR (depth, batch_size, num_queries, num_classes + 1)
        outputs_class = self.class_embedding(query_features)

        # (1, batch_size, num_queries, 2) OR (depth, batch_size, num_queries, 2)
        outputs_segment = self.segment_embedding(query_features)

        # (1, batch_size, max_eseq_length + 1) OR (depth, batch_size, max_eseq_length + 1)
        outputs_count = predict_event_num_with_depth(self.count_head, query_features)

        out = {'pred_logits': outputs_class[-1], 
                'pred_segments': outputs_segment[-1],
                'pred_count': outputs_count[-1],
            }
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment, outputs_count)

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segment, outputs_count, is_enc_aux=False):
        if is_enc_aux:
            return [{'pred_logits': a, 'pred_segments': b, 'pred_count': c}
                for a, b, c in zip(outputs_class, outputs_segment, outputs_count)]
        
        else:
            return [{'pred_logits': a, 'pred_segments': b, 'pred_count': c}
                for a, b, c in zip(outputs_class[:-1], outputs_segment[:-1], outputs_count[:-1])]
    

    @torch.jit.unused
    def _set_aux_loss_caption(self, outputs_caption):
        return [{'pred_captions': a} for a in outputs_caption[:-1]]


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


    def get_segment_features(self, features, denormalized_segments, idx, video_durations):
        """
        Gets features within a specific boundary (based on selected bipartite matching indices) from pre-computed video features
        Parameters:
            features : Tensor of dimension (batch_size, num_tokens, d_model). These are the pre-computed features
            pred_segments : Tensor of dimension (batch_size, num_queries, 2). These are the pre-computed event/segment boundaries.
            indices : matching between the outputs of the last layer and the targets
                    list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))
            video_durations (tensor, float): (batch_size,), representing duration of videos

        Returns:
            pred_features : Tensor of dimension (nb_target_segments, num_tokens, d_model)
            pred_features_src_padding_mask : Tensor of dimension (nb_target_segments, num_tokens)
        """
        
        # idx = get_src_permutation_idx(indices)
        # denormalized_segments = denormalize_segments(pred_segments[idx], video_durations, idx[0])

        pred_features, pred_features_src_padding_mask = self.crop_segments(features, denormalized_segments, idx[0], video_durations)

        return pred_features, pred_features_src_padding_mask


    def crop_segments(self, features, denormalized_segments, segment_batch_id, video_durations):
        """
        Crops the video features within a specific boundary (based on selected bipartite matching indices)
        Parameters:
            features : Tensor of dimension (batch_size, num_tokens, d_model). These are the pre-computed features
            denormalized_segments : Tensor of dimension (nb_target_segments, 2). start time and end time of selected segments
            segment_batch_id (tensor, int): (num_proposals,), representing batch id of corresponding segment
            video_durations (tensor, float): (batch_size,), representing duration of videos

        Returns:
            pred_features : Tensor of dimension (batch_size, max_gt_target_segments, num_tokens, d_model)
            pred_features_src_padding_mask : Tensor of dimension (batch_size, max_gt_target_segments, num_tokens)            
        """

        batch_size, num_tokens, d_model = features.shape

        # normalize segments with respect to duration
        durations_per_proposal = torch.tensor([video_durations[batch_id] for batch_id in segment_batch_id])

        pred_features = torch.zeros([denormalized_segments.shape[0], num_tokens, d_model])
        pred_features_src_padding_mask = torch.ones([denormalized_segments.shape[0], num_tokens], dtype=torch.bool)

        # video_rescale_len = floor((2**(self.num_feature_levels - 1) / (2**self.num_feature_levels - 1)) * num_tokens)

        for n in range(self.num_feature_levels):
            lower_limit = floor(self.video_rescale_len * ((2**n - 1) / 2**(n - 1)))
            upper_limit = floor(self.video_rescale_len * ((2**(n + 1) - 1) / 2**n))
            diff = upper_limit - lower_limit

            start_token = torch.clamp((lower_limit + (diff * denormalized_segments[:, 0] / durations_per_proposal)).round().long(), min=lower_limit, max=upper_limit-1)
            end_token = torch.clamp((lower_limit + (diff * denormalized_segments[:, 1] / durations_per_proposal)).round().long(), min=lower_limit, max=upper_limit-1)

            for i, batch_id in enumerate(segment_batch_id):
                pred_features[i, start_token[i]:end_token[i]] = features[batch_id, start_token[i]:end_token[i], :]
                pred_features_src_padding_mask[i, start_token[i]:end_token[i]] = False

        return pred_features, pred_features_src_padding_mask

    
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
