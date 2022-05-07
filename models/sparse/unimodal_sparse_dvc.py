""" 
DVC model for event segmentation and captioning
"""

import math
from math import floor, ceil
import copy

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
    def __init__(self, input_modalities, num_queries, d_model, num_classes, aux_loss, matcher, threshold, max_eseq_length,
                vocab, seq_len, embedding_matrix, 
                sparse_detr_args, caption_args, use_differentiable_mask=False):
        
        """
        UnimodalSparseDVC model
        """

        super(UnimodalSparseDVC, self).__init__()
        
        self.input_modalities = input_modalities
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.num_classes = num_classes
        self.threshold = threshold

        self.query_embedding = nn.Embedding(num_queries, d_model * 2)

        self.class_embedding_encoder = nn.Linear(d_model, num_classes + 1)
        self.class_embedding_decoder = nn.Linear(d_model, num_classes + 1)

        self.segment_embedding_encoder = FFN(in_dim=d_model, hidden_dim=d_model, out_dim=2, num_layers=3)
        self.segment_embedding_decoder = FFN(in_dim=d_model, hidden_dim=d_model, out_dim=2, num_layers=3)
        
        self.count_head_encoder = nn.Linear(d_model, max_eseq_length + 1)
        self.count_head_decoder = nn.Linear(d_model, max_eseq_length + 1)

        self.matcher = matcher

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

        # Context Module
        self.num_feature_levels = sparse_detr_args.num_feature_levels
        self.video_rescale_len = sparse_detr_args.video_rescale_len
        self.num_tokens = ceil(((2**self.num_feature_levels - 1) / 2**(self.num_feature_levels - 1)) * self.video_rescale_len)
        
        self.use_differentiable_mask = use_differentiable_mask
        if use_differentiable_mask:
            self.context_mask_model = ContextMaskModel(in_dim=(2 + d_model), out_dim=(self.num_tokens))

        # Captioning module
        self.seq_len = seq_len
        self.vocab = vocab
        self.unimodal_caption_decoder = build_unimodal_caption_decoder(caption_args, len(vocab), seq_len, embedding_matrix)
        

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
    
        # Retrieve the matching between the outputs of the last layer and the targets, has torch.no_grad() in forward call
        # list (len=batch_size) of tuple of tensors (tuple dimensions=(2, gt_target_segments))
        indices = self.matcher(out, obj['video_target']) 

        # Context Features
        video_durations = list(obj['video_length'][:, 1])
        idx = get_src_permutation_idx(indices)
        denormalized_segments = denormalize_segments(out['pred_segments'][idx], video_durations, idx[0])

        # (nb_target_segments, num_tokens, d_model), (nb_target_segments, num_tokens)
        memory, memory_mask = self.get_segment_features(memory, denormalized_segments, idx, video_durations)

        memory = memory.to(video.device)

        memory_mask = memory_mask.unsqueeze(1).unsqueeze(1)    # (nb_target_segments, 1, 1, num_tokens)
        memory_mask = memory_mask.to(video.device)

        # Differentiable Mask
        if self.use_differentiable_mask:
            # TODO - use outputs_segment and use [-1] for pred_memory_mask
            # input_to_context_mask = torch.cat([out['pred_segments'], torch.squeeze(query_features)], 2).reshape(batch_size, -1)
            # input_to_context_mask = out['pred_segments'].reshape(batch_size, -1)    # (batch_size, num_queries*2)

            query_features_selected_segments = query_features[-1][idx]  # (nb_target_segments, d_model)

            input_to_context_mask = torch.cat([denormalized_segments.to(video.device), query_features_selected_segments], 1)
            pred_memory_mask = self.context_mask_model(input_to_context_mask)   # (nb_target_segments, num_tokens)

            # Gating mechanism for memory_mask TODO: scores
            seg_confidence = torch.ones([memory.shape[0], 1]).to(video.device)   # (nb_target_segments, 1)

            pred_memory_mask = seg_confidence * pred_memory_mask + (1 - seg_confidence) * torch.squeeze(memory_mask)
                        
            out['pred_memory_mask'] = pred_memory_mask    # last layer value will be returned at the end
            
            assert out['pred_memory_mask'].shape == torch.squeeze(memory_mask).shape

            pred_memory_mask = (pred_memory_mask.sigmoid() > 0.5)
            pred_memory_mask = pred_memory_mask.unsqueeze(1).unsqueeze(1)    # (nb_target_segments, 1, 1, num_tokens)
        
        # Caption Decoder
        if is_training:
            captions = obj['cap_tensor'][:, :-1]    # (total_caption_num, max_caption_length - 1) - <eos> token should be the last predicted token 
            
            padding_mask = obj['cap_mask'][:, :-1]    # (total_caption_num, max_caption_len - 1)

            tgt_mask = self.make_tgt_mask(captions, padding_mask)    # (total_caption_num, 1, max_caption_length - 1, max_caption_length - 1)
            tgt_mask = tgt_mask.to(captions.device)

            # TODO - add pos embed for memory
            # (1, total_caption_num, max_caption_length - 1, vocab_size) OR (caption_decoder_depth, total_caption_num, max_caption_length - 1, vocab_size)
            if self.use_differentiable_mask:
                outputs_caption = self.unimodal_caption_decoder(captions, memory, tgt_mask, padding_mask, pred_memory_mask)
            else:
                outputs_caption = self.unimodal_caption_decoder(captions, memory, tgt_mask, padding_mask, memory_mask)

            out["pred_captions"] = outputs_caption[-1]    # (total_caption_num, max_caption_length - 1, vocab_size)

            outputs_caption_last_layer = torch.argmax(outputs_caption[-1], dim=2)    # (total_caption_num, max_caption_length - 1)
            
            indices_aux = []
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment, outputs_count)
                for i, aux_outputs in enumerate(out['aux_outputs']):
                    indices_aux.append(self.matcher(aux_outputs, obj['video_target']))
                
                out['aux_outputs_caption'] = self._set_aux_loss_caption(outputs_caption)    # caption depth could be different

            if self.use_differentiable_mask:
                return out, outputs_caption_last_layer, indices, indices_aux, torch.squeeze(memory_mask).float()
            else:
                return out, outputs_caption_last_layer, indices, indices_aux, None


        # Inference
        else:
            # Initialize the captions with the `START_TOKEN` and `PAD_TOKEN`    # (total_caption_num, max_caption_length - 1)
            captions = torch.ones([memory.shape[0], self.seq_len - 1], dtype=torch.int32)    # PAD_TOKEN
            captions[:, 0] = self.vocab['<bos>']    # START_TOKEN
            captions = captions.to(memory_mask.device)

            # Loop control Variables
            total_caption_num = memory.shape[0]
            total_caption_done = 0
            caption_done_indices = []
            outputs_captions_val = []

            # Since END_TOKEN can be predicted even if the caption does not reach max_caption_length
            # range(1, max_caption_length - 1)
            for word_index in range(1, self.seq_len - 1):
                captions_padding_mask = self.make_padding_mask(captions)    # (total_caption_num, max_caption_length - 1)
                captions_padding_mask = captions_padding_mask.to(memory_mask.device)
                
                tgt_mask = self.make_tgt_mask(captions, captions_padding_mask)  # (total_caption_num, 1, max_caption_length - 1, max_caption_length - 1)
                tgt_mask = tgt_mask.to(captions.device)

                # (1, total_caption_num, max_caption_length - 1, vocab_size) OR (depth, total_caption_num, max_caption_length - 1, vocab_size)
                if self.use_differentiable_mask:
                    outputs_caption_val = self.unimodal_caption_decoder(captions, memory, tgt_mask, captions_padding_mask, pred_memory_mask)
                else:
                    outputs_caption_val = self.unimodal_caption_decoder(captions, memory, tgt_mask, captions_padding_mask, memory_mask)

                out['pred_captions'] = outputs_caption_val[-1]

                outputs_caption_last_layer = torch.argmax(outputs_caption_val[-1], dim=2)    # (total_caption_num, max_caption_length - 1)
                # print('----------------------', captions[0])
                # print('----------', outputs_caption_last_layer[0])
                # print('--------', captions_padding_mask[0])
                # Update predicted word in captions
                if faster_eval:
                    captions[:, word_index] = outputs_caption_last_layer[:, word_index] # if it doesn't matter whether the predicted token is END_TOKEN
                    # print('------------wdewf', word_index, outputs_caption_last_layer[0, word_index].item())
                    # print('------------', outputs_caption_last_layer[0])
                    # print('------------', captions[0, word_index])
                else:
                    for caption_index in range(total_caption_num):
                        if caption_index not in caption_done_indices:
                            captions[caption_index, word_index] = outputs_caption_last_layer[caption_index, word_index]

                            if outputs_caption_last_layer[caption_index, word_index] == self.vocab['<eos>']:    # if END_TOKEN predicted
                                caption_done_indices.append(caption_index)
                                total_caption_done += 1

                        if total_caption_done == total_caption_num:     # if all captions done
                            break

            if faster_eval:
                # For adding END_TOKEN at the end irrespective of whether it already exists in caption
                end_token = torch.full([captions.shape[0], 1], self.vocab['<eos>'], dtype=torch.int32).to(captions.device)    # `END_TOKEN` (3) column, (total_caption_num, 1)
                captions_with_eos = torch.cat((captions, end_token), 1)  # (total_caption_num, max_caption_length)
            else:
                # Add END_TOKEN or PAD_TOKEN as the last token
                last_token = torch.tensor([self.vocab['<pad>'] if self.vocab['<eos>'] in c else self.vocab['<eos>'] for c in captions], dtype=torch.int32).reshape([-1, 1]).to(captions.device)    # (total_caption_num, 1)
                captions_with_eos = torch.cat((captions, last_token), 1)  # (total_caption_num, max_caption_length)

            # TODO - check use in eval
            indices_aux = []
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment, outputs_count)
                for i, aux_outputs in enumerate(out['aux_outputs']):
                    indices_aux.append(self.matcher(aux_outputs, obj['video_target']))
                
                out['aux_outputs_caption'] = self._set_aux_loss_caption(outputs_caption_val)

            if self.use_differentiable_mask:
                return out, captions_with_eos, indices, indices_aux, torch.squeeze(memory_mask).float()
            else:
                return out, captions_with_eos, indices, indices_aux, None
            

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


    def make_tgt_mask(self, target, tgt_padding_mask):
        """
        Generates a mask that is a combination of a lookahead mask and a padding mask
        
        Parameters:
            target (Tensor): Tensor of dimension (batch_size, seq_len)
            tgt_padding_mask (Tensor): Padding mask of dimension (batch_size, seq_len)
        
        Returns:
            tgt_mask (Tensor): Tensor of dimention (batch_size, 1, seq_len, seq_len)
        """

        batch_size, seq_len = target.shape

        look_ahead_mask = 1 - torch.tril(torch.ones((seq_len, seq_len)))
        look_ahead_mask = look_ahead_mask.to(tgt_padding_mask.device)

        tgt_mask = torch.maximum(tgt_padding_mask.unsqueeze(1).unsqueeze(1), look_ahead_mask).bool()

        return tgt_mask    # (batch_size, 1, seq_len, seq_len)


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
            

    def load_weights(self, model_official):

        """
        Loads the weights and biases from the pre-trained model to the current model for modules in the UnimodalSparseDVC model
        These weights include positional embeddings.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
        """

        load_positional_embeddings(self, model_official)