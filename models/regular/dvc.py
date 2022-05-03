""" 
DVC model for event segmentation and captioning
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .vivit import build_vivit
from .decoder import build_decoder
from ..unimodal_caption_decoder import build_unimodal_caption_decoder

from ..modules.embedding_layers import PositionalEmbedding
from ..modules.layers import FFN

from ..load_weights import load_positional_embeddings

from utils.preds_postprocess import get_src_permutation_idx, denormalize_segments

# TODO - src mask
# TODO - check devices for tensors
class DVC(nn.Module):
    def __init__(self, num_queries, d_model, num_classes, aux_loss, matcher, vocab_size, seq_len, embedding_matrix, 
                vivit_args, ast_args, decoder_args, caption_args):
        
        """
        DVC type model
        """

        super(DVC, self).__init__()
        
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        self.query_embedding = nn.Embedding(num_queries, d_model)

        self.class_embedding = nn.Linear(d_model, num_classes + 1)
        self.segment_embedding = FFN(in_dim=d_model, hidden_dim=d_model, out_dim=2, num_layers=3)

        self.matcher = matcher
        
        # for ViViT    
        self.positional_embedding_layer = None
        self.spatial_positional_embedding_layer = None

        if vivit_args.model_name == 'spatio temporal attention':
            self.positional_embedding_layer = PositionalEmbedding((1, vivit_args.num_frames * vivit_args.num_patches + 1, d_model), vivit_args.positional_embedding_dropout) 
            
        elif vivit_args.model_name == 'factorised encoder':
            self.spatial_positional_embedding_layer = PositionalEmbedding((1, vivit_args.num_patches + 1, d_model), vivit_args.positional_embedding_dropout)
            self.positional_embedding_layer = PositionalEmbedding((1, vivit_args.num_frames + 1, d_model), vivit_args.positional_embedding_dropout)

        else:
            self.positional_embedding_layer = PositionalEmbedding((1, vivit_args.num_frames, vivit_args.num_patches, d_model), vivit_args.positional_embedding_dropout)


        self.vivit = build_vivit(vivit_args)

        # TODO - add bimodal encoder
        self.decoder = build_decoder(decoder_args)
        
        # Captioning module
        self.caption_decoder = build_caption_decoder(caption_args, vocab_size, seq_len, embedding_matrix)
        

        # if weight_load and model_official is not None:
        #     self.load_weights(model_official)

        # elif weight_init:
        #     self.init_weights()



    # TODO - use log softmax?
    # TODO - padding and src_mask for vid features as input to caption decoder  
    # TODO - add position embedding in caption decoder
    def forward(self, obj, is_training=True, faster_eval=False):

        """
        Performs a forward pass on the DVC model which consists of the encoders, proposal decoder and caption decoder
  
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

        video_input = obj['video_tensor']    # (batch_size, in_channels, num_frames, img_size, img_size)
        
        # Encoder
        # (batch_size, num_frames * num_patches + 1, d_model) OR
        # (batch_size, num_frames + 1, d_model) OR 
        # (batch_size, num_frames, num_patches, d_model) 
        video = self.vivit(video_input, self.positional_embedding_layer, self.spatial_positional_embedding_layer)

        # TODO - check grad later
        if self.vivit.model_name == 'factorised self attention' or self.vivit.model_name == 'factorised dot product attention':
            video = video.reshape(video.shape[0], -1, video.shape[-1])


        # Decoder
        query_embedding_weight = self.query_embedding.weight.unsqueeze(0).repeat(video_input.shape[0], 1, 1)    # (batch_size, num_queries, d_model)
        target = torch.zeros_like(query_embedding_weight)

        # (1, batch_size, num_queries, d_model) OR # (depth, batch_size, num_queries, d_model)
        res = self.decoder(target=target, memory=video, 
                        positional_embedding_layer=self.positional_embedding_layer, query_embedding=query_embedding_weight, mask=None)


        # (1, batch_size, num_queries, num_classes + 1) OR (depth, batch_size, num_queries, num_classes + 1)
        outputs_class = self.class_embedding(res).softmax(dim=-1)

        # (1, batch_size, num_queries, 2) OR (depth, batch_size, num_queries, 2)
        outputs_segment = self.segment_embedding(res).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_segment[-1]}


        # Retrieve the matching between the outputs of the last layer and the targets
        # list (len=batch_size) of tuple of tensors (tuple dimensions=(2, gt_target_segments))
        indices = self.matcher(out, obj['video_target']) 

        # Context Features
        video_durations = list(obj['video_length'][:, 1])

        # (nb_target_segments, num_tokens, d_model), (nb_target_segments, num_tokens)
        memory, memory_mask = self.get_segment_features(video, out['pred_segments'], indices, video_durations)

        memory = memory.to(video.device)
        # memory.requires_grad = True

        memory_mask = memory_mask.unsqueeze(1).unsqueeze(1)    # (nb_target_segments, 1, 1, num_tokens)
        memory_mask = memory_mask.to(video.device)
        
        # Caption Decoder
        if is_training:
            captions = obj['cap_tensor'][:, :-1]    # (total_caption_num, max_caption_length - 1) - <eos> token should be the last predicted token 
            tgt_mask = self.make_tgt_mask(captions, obj['cap_mask'][:, :-1])    # (total_caption_num, 1, max_caption_length - 1, max_caption_length - 1)
            tgt_mask = tgt_mask.to(captions.device)
        
            # (1, total_caption_num, max_caption_length - 1, vocab_size) OR (depth, total_caption_num, max_caption_length - 1, vocab_size)
            outputs_captions = self.caption_decoder(captions, memory, nn.Identity(), tgt_mask, memory_mask)

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


    def get_segment_features(self, features, pred_segments, indices, video_durations):
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
        
        idx = get_src_permutation_idx(indices)
        denormalized_segments = denormalize_segments(pred_segments[idx], video_durations, idx[0])

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

        start_token = torch.clamp((num_tokens * denormalized_segments[:, 0] / durations_per_proposal).round().long(), min=0, max=num_tokens-1)
        end_token = torch.clamp((num_tokens * denormalized_segments[:, 1] / durations_per_proposal).round().long(), min=0, max=num_tokens-1)

        pred_features = torch.zeros([denormalized_segments.shape[0], num_tokens, d_model])
        pred_features_src_padding_mask = torch.zeros([denormalized_segments.shape[0], num_tokens])

        for i, batch_id in enumerate(segment_batch_id):
            pred_features[i, start_token[i]:end_token[i]] = features[batch_id, start_token[i]:end_token[i], :]
            pred_features_src_padding_mask[i, start_token[i]:end_token[i]] = 1

        return pred_features, pred_features_src_padding_mask

    
    def init_weights(self):

        """
        Initialises the weights and biases of the modules in the DVC model.
        These parameters include positional embeddings.
        """

        trunc_normal_(self.positional_embedding_layer.positional_embedding, std=.02)
        if self.vivit.model_name == 'factorised encoder':
            trunc_normal_(self.spatial_positional_embedding_layer.positional_embedding, std=.02)
            

    def load_weights(self, model_official):

        """
        Loads the weights and biases from the pre-trained model to the current model for modules in the DVC model
        These weights include positional embeddings.

        Parameters:
            `model_custom`: The current ViViT model
            `model_official`: The model which would be used to load the pre-trained weights
        """

        load_positional_embeddings(self, model_official)