""" 
DVC model for event segmentation and captioning
"""

import math
from math import floor, ceil

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .decoder import build_decoder
from .encoder import build_encoder
from ..unimodal_caption_decoder import build_unimodal_caption_decoder

from ..modules.embedding_layers import PositionalEncoding, PositionEmbeddingVideoSine
from ..modules.layers import FFN, ContextMaskModel
from ..modules.misc_modules import NestedTensor, predict_event_num_with_depth

from ..load_weights import load_positional_embeddings

from utils.preds_postprocess import get_src_permutation_idx, denormalize_segments

# TODO - src mask
# TODO - check devices for tensors
class DVC(nn.Module):
    def __init__(self, input_modalities, num_queries, d_model, num_classes, aux_loss, max_eseq_length, 
                    encoder_args, decoder_args):
        
        """
        DVC type model
        """

        super(DVC, self).__init__()
        
        self.input_modalities = input_modalities
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.num_classes = num_classes

        self.query_embedding = nn.Embedding(num_queries, d_model)

        self.class_embedding= nn.Linear(d_model, num_classes)
        self.segment_embedding = FFN(in_dim=d_model, hidden_dim=d_model, out_dim=2, num_layers=3)
        self.count_head = nn.Linear(d_model, max_eseq_length + 1)
        
        assert 'video' in input_modalities or 'audio' in input_modalities, f'input_modalities should contain one of "video" or "audio". You have {input_modalities}'

        self.positional_embedding = PositionalEncoding(d_model, dropout=encoder_args.positional_embedding_dropout)

        # TODO - add bimodal encoder
        self.encoder = build_encoder(encoder_args)

        self.decoder = build_decoder(decoder_args)

        nn.init.constant_(self.segment_embedding.layers[-1].weight.data, 0.)
        nn.init.constant_(self.segment_embedding.layers[-1].bias.data[:2], 0.)
        nn.init.constant_(self.segment_embedding.layers[-1].bias.data[2:], -2.0)
        

        # self.init_weights()


    # TODO - use log softmax?
    # TODO - padding and src_mask for vid features as input to caption decoder  
    # TODO - add position embedding in caption decoder
    def forward(self, obj):

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

        video = obj['video_tensor']    # (batch_size, num_tokens_v, d_model)
        video_mask = obj['video_mask']    # (batch_size, num_tokens_v)
        
        durations = obj['video_length'][:, 1]   # (batch_size)

        # audio = obj['audio_tensor']    # (batch_size, num_tokens_a, d_model)
        # audio_mask = obj['audio_mask']    # (batch_size, num_tokens_a)
        
        batch_size, _, _ = video.shape

        memory = self.encoder(src=video, src_positional_embedding=self.positional_embedding, 
                                src_mask=None, src_padding_mask=video_mask)[0]    # (batch_size, num_tokens, d_model)

        # vf = video.transpose(1, 2)  # (batch_size, num_tokens, d_model) --> (batch_size, d_model, num_tokens)
        # vf_nt = NestedTensor(vf, video_mask, durations)
        # positional_embedding = self.pos_embed(vf_nt).transpose(1, 2)

        query_embedding_weight = self.query_embedding.weight
        query_embed = query_embedding_weight.unsqueeze(0).expand(batch_size, -1, -1)    # (batch_size, num_queries, d_model)  

        # (1, batch_size, num_queries, d_model) OR # (depth, batch_size, num_queries, d_model)
        query_features = self.decoder(memory=memory, memory_positional_embedding=self.positional_embedding, query_embedding=query_embed, 
                                    tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=video_mask)


        # (1, batch_size, num_queries, num_classes + 1) OR (depth, batch_size, num_queries, num_classes + 1)
        outputs_class = self.class_embedding(query_features)

        # (1, batch_size, num_queries, 2) OR (depth, batch_size, num_queries, 2)
        outputs_segment = self.segment_embedding(query_features)

        # (1, batch_size, max_eseq_length + 1) OR (depth, batch_size, max_eseq_length + 1)
        outputs_count = predict_event_num_with_depth(self.count_head, query_features)
        
        out = {'pred_logits': outputs_class[-1], 'pred_count': outputs_count[-1], 'pred_segments': outputs_segment[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment, outputs_count)

        return out

    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segment, outputs_count):
        return [{'pred_logits': a, 'pred_segments': b, 'pred_count': c}
            for a, b, c in zip(outputs_class, outputs_segment, outputs_count)]
    
    def init_weights(self):

        """
        Initialises the weights and biases of the modules in the DVC model.
        These parameters include positional embeddings.
        """

        trunc_normal_(self.positional_embedding_layer.positional_embedding, std=.02)
        if self.vivit.model_name == 'factorised encoder':
            trunc_normal_(self.spatial_positional_embedding_layer.positional_embedding, std=.02)
