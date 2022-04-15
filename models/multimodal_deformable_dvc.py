""" 
DVC model for event segmentation and captioning
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .vivit import build_vivit
from .ast import build_ast
from .multimodal_deformable_transformer import build_multimodal_deformable_transformer
from .base_encoder import build_base_encoder
from .caption_decoder import build_caption_decoder

from .modules.embedding_layers import PositionalEmbedding
from .modules.misc_modules import decide_two_stage
from .modules.layers import FFN

from .load_weights import load_positional_embeddings


# TODO - src mask
# TODO - check devices for tensors
class MultimodalDeformableDVC(nn.Module):
    def __init__(self, input_modalities,num_queries, d_model, num_classes, aux_loss, matcher, 
                vocab_size, seq_len, embedding_matrix, 
                vivit_args, ast_args, detr_args, caption_args):
        
        """
        MultimodalDeformableDVC model
        """

        super(MultimodalDeformableDVC, self).__init__()
        
        self.input_modalities = input_modalities
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        self.query_embedding = nn.Embedding(num_queries, d_model * 2)

        self.class_embedding = nn.Linear(d_model, num_classes + 1)
        self.segment_embedding = FFN(in_dim=d_model, hidden_dim=d_model, out_dim=2, num_layers=3)

        self.matcher = matcher

        assert 'video' in input_modalities and 'audio' in input_modalities, f'input_modalities should contain both, "video" and "audio". You have {input_modalities}'

        self.base_encoder = build_base_encoder(detr_args)

        # Multimodal Deformable DETR
        self.multimodal_deformable_transformer = build_multimodal_deformable_transformer(detr_args)
        
        # Captioning module
        self.caption_decoder = build_caption_decoder(caption_args, vocab_size, seq_len, embedding_matrix)
        

        # if weight_load and model_official is not None:
        #     self.load_weights(model_official)

        # elif weight_init:
        #     self.init_weights()



    # TODO - use log softmax?
    # TODO - padding and src_mask for vid features as input to caption decoder  
    # TODO - add position embedding in caption decoder
    # TODO - check all pos embed
    def forward(self, obj, is_training=True, faster_eval=False):

        """
        Performs a forward pass on the MultimodalDeformableDVC model which consists of the encoders, proposal decoder and caption decoder
  
        Parameters:
            obj (collections.defaultdict): Consisitng of various keys including 
                                           video_tensor (batch_size, in_channels, num_frames, img_size, img_size)
                                           video_mask (batch_size, num_frames)
                                           video_length (batch_size, 3) - num_frames, duration, gt_target_segments
        
        Returns:
            out (dictionary) : It returns a dict with the following elements:
                                - "pred_logits": the classification logits (including no-object) for all queries
                                                    shape (batch_size, num_queries, num_classes+ 1)
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

        audio = obj['audio_tensor']    # (batch_size, num_tokens_a, d_model)
        audio_mask = obj['audio_mask']    # (batch_size, num_tokens_a)
        
        batch_size, _, _ = video.shape

        # Base Encoder - for multi-scale features
        video_srcs, video_masks, video_pos = self.base_encoder(video, video_mask, durations, 'video')
        audio_srcs, audio_masks, audio_pos = self.base_encoder(audio, audio_mask, durations, 'audio')

        # Forword Encoder
        video_src_flatten, video_temporal_shapes, video_level_start_index, video_valid_ratios, video_lvl_pos_embed_flatten, video_mask_flatten = self.multimodal_deformable_transformer.prepare_encoder_inputs(video_srcs, video_masks, video_pos)
        audio_src_flatten, audio_temporal_shapes, audio_level_start_index, audio_valid_ratios, audio_lvl_pos_embed_flatten, audio_mask_flatten = self.multimodal_deformable_transformer.prepare_encoder_inputs(audio_srcs, audio_masks, audio_pos)

        # (batch_size, sum of num_tokens in all levels, d_model) - Multi-scale frame features
        video_memory, audio_memory = self.multimodal_deformable_transformer.forward_encoder(video_src_flatten, video_temporal_shapes, 
                                                                video_level_start_index, video_valid_ratios, 
                                                                video_lvl_pos_embed_flatten, video_mask_flatten, 
                                                                audio_src_flatten, audio_temporal_shapes, 
                                                                audio_level_start_index, audio_valid_ratios, 
                                                                audio_lvl_pos_embed_flatten, audio_mask_flatten)


        # Forword Decoder
        # TODO - see transformer_input_type = "gt_proposals"
        transformer_input_type = "queries"
        gt_boxes = None
        gt_boxes_mask = None
        criterion = None
        two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type, gt_boxes, gt_boxes_mask, criterion)

        if two_stage:
            init_reference, tgt, reference_points, query_embedding = self.multimodal_deformable_transformer.prepare_decoder_input_proposal(proposals)
        else:
            query_embedding_weight = self.query_embedding.weight
            proposals_mask = torch.ones(batch_size, query_embedding_weight.shape[0], device=query_embedding_weight.device).bool()  #   (batch_size, num_queries)
            init_reference, tgt, reference_points, query_embedding_weight = self.multimodal_deformable_transformer.prepare_decoder_input_query(batch_size, query_embedding_weight)


        # query_features (depth, batch_size, num_queries, d_model)
        # inter_reference = (depth, batch_size, num_queries, 1)
        query_features, inter_references = self.multimodal_deformable_transformer.forward_decoder(tgt, reference_points, query_embedding_weight, proposals_mask, video_memory, video_temporal_shapes,
                                                        video_level_start_index, video_valid_ratios,
                                                        video_mask_flatten, audio_memory, audio_temporal_shapes,
                                                        audio_level_start_index, audio_valid_ratios,
                                                        audio_mask_flatten,  disable_iterative_refine)


        # (1, batch_size, num_queries, num_classes + 1) OR (depth, batch_size, num_queries, num_classes + 1)
        outputs_class = self.class_embedding(query_features).softmax(dim=-1)

        # (1, batch_size, num_queries, 2) OR (depth, batch_size, num_queries, 2)
        outputs_segment = self.segment_embedding(query_features).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_segment[-1]}


        # Retrieve the matching between the outputs of the last layer and the targets
        # list (len=batch_size) of tuple of tensors (tuple dimensions=(2, gt_target_segments))
        indices = self.matcher(out, obj['video_target']) 
        
        # Context Features
        with torch.no_grad():
            max_gt_target_segments = obj['gt_segments'].shape[1]

            # (nb_target_segments, num_tokens, d_model), (nb_target_segments, num_tokens)
            memory, memory_mask = self.get_segment_features(video, out['pred_segments'], indices, max_gt_target_segments)

        memory = memory.to(video.device)
        memory.requires_grad = True

        memory_mask = memory_mask.unsqueeze(1).unsqueeze(1)    # (nb_target_segments, 1, 1, num_tokens)
        memory_mask = memory_mask.to(video.device)
        
        # Caption Decoder
        if is_training:
            captions = obj['cap_tensor'][:, :-1]    # (total_caption_num, max_caption_length - 1) - <eos> token should be the last predicted token 
            
            padding_mask = obj['cap_mask'][:, :-1]    # (total_caption_num, max_caption_len - 1)

            tgt_mask = self.make_tgt_mask(captions, padding_mask)    # (total_caption_num, 1, max_caption_length - 1, max_caption_length - 1)
            tgt_mask = tgt_mask.to(captions.device)
        
            # (1, total_caption_num, max_caption_length - 1, vocab_size) OR (depth, total_caption_num, max_caption_length - 1, vocab_size)
            outputs_captions = self.caption_decoder(captions, memory, nn.Identity(), tgt_mask, padding_mask, memory_mask)

            out["pred_captions"] = outputs_captions[-1]    # (total_caption_num, max_caption_length - 1, vocab_size)

            # TODO - indices for aux loss
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment, outputs_captions)

            return out, indices

        else:   # Inference
            # Initialize the captions with the `START_TOKEN` and `PAD_TOKEN`    # (total_caption_num, max_caption_length-1)
            captions = torch.ones([memory.shape[0], 19], dtype=torch.int32)    # `PAD_TOKEN`
            captions[:, 0] = 2  # `START_TOKEN`
            captions = captions.to(memory_mask.device)  # TODO Temporary

            # Loop control Variables
            total_caption_num = memory.shape[0]
            total_caption_done = 0
            caption_done_indices = []

            # Since END_TOKEN can be predicted even if the caption not reaches max_caption_length
            # range(1, max_caption_length-1)
            for word_index in range(1, 19):
                captions_padding_mask = self.make_padding_mask(captions)
                captions_padding_mask = captions_padding_mask.to(memory_mask.device)    # TODO Temporary
                
                tgt_mask = self.make_tgt_mask(captions, captions_padding_mask)  # (total_caption_num, 1, max_caption_length - 1, max_caption_length - 1)
                tgt_mask = tgt_mask.to(captions.device)  # TODO Temporary

                # (1, total_caption_num, max_caption_length - 1, vocab_size)
                outputs_captions = self.caption_decoder(captions, memory, nn.Identity(), tgt_mask, memory_mask)
                outputs_captions = torch.argmax(outputs_captions[-1], dim=2)    # (total_caption_num, max_caption_length-1)

                # Update predicted word in captions
                if faster_eval:
                    captions[:, word_index] = outputs_captions[:, word_index] # if it doesn't matter whether the predicted token is END_TOKEN

                else:
                    for caption_index in range(total_caption_num):
                        if caption_index not in caption_done_indices:
                            captions[caption_index, word_index] = outputs_captions[caption_index, word_index]

                            if outputs_captions[caption_index, word_index] == 3:    # if END_TOKEN predicted
                                caption_done_indices.append(caption_index)
                                total_caption_done += 1

                        if total_caption_done == total_caption_num:     # if all captions done
                            break

            if faster_eval:
                # For adding END_TOKEN at the end irrespective of whether it already exists in caption
                end_token = torch.full([captions.shape[0], 1], 3, dtype=torch.int32).to(captions.device)    # `END_TOKEN` (3) column, (total_caption_num, 1)
                captions = torch.cat((captions, end_token), 1)  # (total_caption_num, max_caption_length)
            else:
                # Add END_TOKEN or PAD_TOKEN as the last token
                last_token = torch.tensor([1 if 3 in c else 3 for c in captions], dtype=torch.int32).reshape([-1, 1]).to(captions.device)    # (total_caption_num, 1)
                captions = torch.cat((captions, last_token), 1)  # (total_caption_num, max_caption_length)

            return out['pred_segments'], captions, out['pred_logits'], indices


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segment, outputs_captions):
        return [{'pred_logits': a, 'pred_segments': b, 'pred_captions': c}
                for a, b, c in zip(outputs_class[:-1], outputs_segment[:-1], outputs_captions[:-1])]

    
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
        look_ahead_mask = torch.tril(torch.ones((seq_len, seq_len))).to(tgt_padding_mask.device)
        tgt_mask = torch.minimum(tgt_padding_mask.unsqueeze(1).unsqueeze(1), look_ahead_mask)
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

        tgt_padding_mask = (target != 1)  # 1 is PAD_TOKEN
        return tgt_padding_mask


    # TODO - make more efficient
    def get_segment_features(self, features, pred_segments, indices, max_gt_target_segments):

        """
        Gets features within a specific boundary (based on selected bipartite matching indices) from pre-computed video features
        Parameters:
            features : Tensor of dimension (batch_size, num_tokens, d_model). These are the pre-computed features
            pred_segments : Tensor of dimension (batch_size, num_queries, 2). These are the pre-computed event/segment boundaries.
            indices : matching between the outputs of the last layer and the targets
                    list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))
            max_gt_target_segments (int): Maximum number of ground truth events/segments in a single video in a batch

        Returns:
            pred_features : Tensor of dimension (nb_target_segments, num_tokens, d_model)
            pred_features_src_padding_mask : Tensor of dimension (nb_target_segments, num_tokens)
        """
        
        batch_size, num_tokens, d_model = features.shape

        pred_segment_boundaries = torch.zeros(batch_size, max_gt_target_segments, 2)

        for i, (pred_idx, _) in enumerate(indices):
            pred_segment_boundaries[i, :pred_idx.shape[0]] = pred_segments[i, pred_idx.long()]
        
        # (batch_size, max_gt_target_segments, num_tokens, d_model) AND (batch_size, max_gt_target_segments, num_tokens) AND (batch_size, max_gt_target_segments)
        pred_features, pred_features_src_padding_mask, pred_segments_padding_mask = self.crop_segments(features, pred_segment_boundaries, indices, max_gt_target_segments)
        
        pred_features = pred_features.reshape(-1, num_tokens, d_model)
        pred_features_src_padding_mask = pred_features_src_padding_mask.reshape(-1, num_tokens)
        pred_segments_padding_mask = pred_segments_padding_mask.reshape(-1)
        
        # removes extra captions (padding) added to satisfy dimension constraints of tensors
        pred_features = pred_features[pred_segments_padding_mask == True]    # (nb_target_segments, num_tokens, d_model)
        pred_features_src_padding_mask = pred_features_src_padding_mask[pred_segments_padding_mask == True]    # (nb_target_segments, num_tokens)
        
        return pred_features, pred_features_src_padding_mask


    # TODO - padding like in BMT??
    def crop_segments(self, features, pred_segment_boundaries, indices, max_gt_target_segments):

        """
        Crops the video features within a specific boundary (based on selected bipartite matching indices)
        Parameters:
            features : Tensor of dimension (batch_size, num_tokens, d_model). These are the pre-computed features
            pred_segment_boundaries : Tensor of dimension (batch_size, max_gt_target_segments, 2). These are the pre-computed event/segment boundaries.
            indices : matching between the outputs of the last layer and the targets
                    list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))
            max_gt_target_segments (int): Maximum number of ground truth events/segments in a single video in a batch

        Returns:
            pred_features : Tensor of dimension (batch_size, max_gt_target_segments, num_tokens, d_model)
            pred_features_src_padding_mask : Tensor of dimension (batch_size, max_gt_target_segments, num_tokens)
            pred_segments_padding_mask : Tensor of dimension (batch_size, max_gt_target_segments)
            
        """

        batch_size, num_tokens, d_model = features.shape
        
        start_quantile = pred_segment_boundaries[:, :, 0]    # (batch_size, max_gt_target_segments)
        end_quantile = pred_segment_boundaries[:, :, 1]    # (batch_size, max_gt_target_segments)
        start_idx = (num_tokens * start_quantile).long().reshape(-1)    # (batch_size * max_gt_target_segments)
        end_idx = (num_tokens * end_quantile).long().reshape(-1)    # (batch_size * max_gt_target_segments)

        for i, (start, end) in enumerate(zip(start_idx, end_idx)):
            if start >= end:
                if start >= num_tokens:
                    start = num_tokens - 1
                    start_idx[i] = start
                    end_idx[i] = num_tokens
                elif end >= num_tokens:
                    end_idx[i] = num_tokens
                else:
                    end_idx[i] = start + 1
        
        start_idx = start_idx.reshape(batch_size, max_gt_target_segments) 
        end_idx = end_idx.reshape(batch_size, max_gt_target_segments)
            
        pred_features = torch.zeros(batch_size, max_gt_target_segments, num_tokens, d_model)
        pred_features_src_padding_mask = torch.zeros(batch_size, max_gt_target_segments, num_tokens)
        pred_segments_padding_mask = torch.zeros(batch_size, max_gt_target_segments, dtype=torch.bool)

        for i in range(batch_size):
            gt_target_segments = len(indices[i][0])
            for j in range(gt_target_segments):
                pred_features[i, j, start_idx[i, j]:end_idx[i, j]] = features[i, start_idx[i, j]:end_idx[i, j], :]
                pred_features_src_padding_mask[i, j, start_idx[i, j]:end_idx[i, j]] = 1
                pred_segments_padding_mask[i, j] = True

        return pred_features, pred_features_src_padding_mask, pred_segments_padding_mask 

    
    def init_weights(self):

        """
        Initialises the weights and biases of the modules in the MultimodalDeformableDVC model.
        These parameters include positional embeddings.
        """

        # trunc_normal_(self.positional_embedding_layer.positional_embedding, std=.02)
        # if self.vivit.model_name == 'factorised encoder':
        #     trunc_normal_(self.spatial_positional_embedding_layer.positional_embedding, std=.02)
        pass
            

    def load_weights(self, model_official):

        """
        Loads the weights and biases from the pre-trained model to the current model for modules in the MultimodalDeformableDVC model
        These weights include positional embeddings.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
        """

        load_positional_embeddings(self, model_official)