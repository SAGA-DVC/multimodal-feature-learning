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

from .unimodal_deformable_transformer import build_unimodal_deformable_transformer
from ..base_encoder import build_base_encoder
from ..unimodal_caption_decoder import build_unimodal_caption_decoder

from ..modules.embedding_layers import PositionEmbeddingVideoSine
from ..modules.layers import FFN, ContextMaskModel
from ..modules.misc_modules import decide_two_stage, inverse_sigmoid, predict_event_num_with_depth

from ..load_weights import load_positional_embeddings

from utils.preds_postprocess import get_src_permutation_idx, denormalize_segments



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CapUnimodalDeformableDVC(nn.Module):
    def __init__(self, input_modalities, num_queries, d_model, num_classes, aux_loss, threshold, max_eseq_length,
                vocab, seq_len, embedding_matrix, detr_args, caption_args):
        
        """
        UnimodalDeformableDVC for captioning
        """

        super(CapUnimodalDeformableDVC, self).__init__()
        
        self.input_modalities = input_modalities
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.threshold = threshold

        self.query_embedding = nn.Embedding(num_queries, d_model * 2)

        assert 'video' in input_modalities or 'audio' in input_modalities, f'input_modalities should contain one of "video" or "audio". You have {input_modalities}'

        self.pos_embed = PositionEmbeddingVideoSine(d_model//2, normalize=True)

        self.base_encoder = build_base_encoder(detr_args)

        # TODO - return intermediate=False deos not output depth dimesntion (dim 0)
        # Unimodal Deformable DETR
        self.unimodal_deformable_transformer = build_unimodal_deformable_transformer(detr_args)

        # Context Module
        self.num_feature_levels = detr_args.num_feature_levels
        self.video_rescale_len = detr_args.video_rescale_len
        self.num_tokens = ceil(((2**self.num_feature_levels - 1) / 2**(self.num_feature_levels - 1)) * self.video_rescale_len)

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

        out = {}

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

        # Context Features
        gt_segments = torch.cat([target['segments'] for target in obj['video_target']], dim=0)    # (nb_target_segments, 2)
        
        segment_batch_id_arr = []
        for i, target in enumerate(obj['video_target']):
            segment_batch_id_arr += [i for j in range(len(target['segments']))]
        
        # segment_batch_id_arr = [i for j in range(len(target['segments'])) for i, target in enumerate(obj['video_target'])]

        segment_batch_id = torch.LongTensor(segment_batch_id_arr)    # (nb_target_segments)

        video_durations = list(obj['video_length'][:, 1])

        denormalized_segments = denormalize_segments(gt_segments, video_durations, segment_batch_id)

        # (nb_target_segments, num_tokens, d_model), (nb_target_segments, num_tokens)
        memory, memory_mask = self.get_segment_features(memory, denormalized_segments, segment_batch_id, video_durations)

        memory = memory.to(video.device)
        memory_mask = memory_mask.to(video.device)    # (nb_target_segments, num_tokens)
        
        # Caption Decoder
        if is_training:
            tgt_captions = obj['cap_tensor'][:, :-1]    # (total_caption_num, max_caption_length - 1) - <eos> token should be the last predicted token 
            
            tgt_padding_mask = obj['cap_mask'][:, :-1]    # (total_caption_num, max_caption_len - 1)

            tgt_mask = self.make_tgt_mask(tgt_captions, tgt_padding_mask.device)    # (max_caption_length - 1, max_caption_length - 1)

            # (1, total_caption_num, max_caption_length - 1, vocab_size) OR (caption_decoder_depth, total_caption_num, max_caption_length - 1, vocab_size)
            outputs_caption = self.unimodal_caption_decoder(tgt=tgt_captions, memory=memory, tgt_mask=tgt_mask, memory_mask=None, tgt_padding_mask=tgt_padding_mask, memory_padding_mask=memory_mask)

            out["pred_captions"] = outputs_caption[-1]    # (total_caption_num, max_caption_length - 1, vocab_size)

            outputs_caption_last_layer = torch.argmax(outputs_caption[-1], dim=2)    # (total_caption_num, max_caption_length - 1)
            
            if self.aux_loss:
                out['aux_outputs_caption'] = self._set_aux_loss_caption(outputs_caption)    # caption depth could be different

            return out, outputs_caption_last_layer


        # Inference
        else:
            # Initialize the captions with the `START_TOKEN` and `PAD_TOKEN`    # (total_caption_num, max_caption_length - 1)
            captions = torch.ones([memory.shape[0], self.seq_len], dtype=torch.int32)    # PAD_TOKEN
            captions[:, 0] = self.vocab['<bos>']    # START_TOKEN
            captions = captions.to(memory_mask.device)

            # Loop control Variables
            total_caption_num = memory.shape[0]
            total_caption_done = 0
            caption_done_indices = []
            outputs_captions_val = []

            # Since END_TOKEN can be predicted even if the caption does not reach max_caption_length
            # range(1, max_caption_length - 1)
            for word_index in range(1, self.seq_len):
                tgt_padding_mask = self.make_padding_mask(captions)    # (total_caption_num, max_caption_length - 1)
                tgt_padding_mask = tgt_padding_mask.to(memory_mask.device)
                
                tgt_mask = self.make_tgt_mask(captions, tgt_padding_mask.device)  # (max_caption_length - 1, max_caption_length - 1)

                # (1, total_caption_num, max_caption_length - 1, vocab_size) OR (depth, total_caption_num, max_caption_length - 1, vocab_size)
                outputs_caption_val = self.unimodal_caption_decoder(tgt=captions, memory=memory, tgt_mask=tgt_mask, memory_mask=None, tgt_padding_mask=tgt_padding_mask, memory_padding_mask=memory_mask)

                out['pred_captions'] = outputs_caption_val[-1]

                outputs_caption_last_layer = torch.argmax(outputs_caption_val[-1], dim=2)    # (total_caption_num, max_caption_length - 1)
                
                # Update predicted word in captions
                if faster_eval:
                    captions[:, word_index] = outputs_caption_last_layer[:, word_index] # if it doesn't matter whether the predicted token is END_TOKEN
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

            if self.aux_loss:
                out['aux_outputs_caption'] = self._set_aux_loss_caption(outputs_caption_val)

            return out, captions_with_eos
            

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


    def get_segment_features(self, features, denormalized_segments, segment_batch_id, video_durations):
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

        pred_features, pred_features_src_padding_mask = self.crop_segments(features, denormalized_segments, segment_batch_id, video_durations)

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
