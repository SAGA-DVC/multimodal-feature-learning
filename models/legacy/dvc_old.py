""" 
DVC model
"""

import torch
import torch.nn as nn

from .caption_decoder import CaptionDecoder
from .transformer import Transformer
from .modules import FFN

# TODO - src mask
class DVC(nn.Module):
    def __init__(self, model_name, num_frames_in, img_size=224, spatial_patch_size=16, temporal_patch_size=1,
                tokenization_method='central frame', in_channels=3, d_model=768, 
                vocab_size=1000, seq_len=10, embedding_matrix=None, emb_weights_req_grad=False,
                depth=12, temporal_depth=4, num_heads=12, 
                mlp_ratio=4., qkv_bias=True, positional_embedding_dropout=0., attention_dropout=0., 
                projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True, classification_head=False, 
                num_classes=None, num_queries=100, aux_loss=False,
                return_preclassifier=True, return_prelogits=False, weight_init=False, weight_load=False, model_official=None,
                return_intermediate=False, matcher=None):
        
        """
        DVC type model
        """

        super(DVC, self).__init__()
        
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        self.query_embedding = nn.Embedding(num_queries, d_model)
        self.class_embedding = nn.Linear(d_model, num_classes + 1)
        self.segment_embedding = FFN(in_dim=d_model, hidden_dim=d_model, out_dim=2, num_layers=3)

        self.transformer = Transformer(model_name=model_name, 
                        num_frames_in=num_frames_in,  
                        img_size=img_size, 
                        spatial_patch_size=spatial_patch_size, 
                        temporal_patch_size=temporal_patch_size,
                        tokenization_method=tokenization_method, 
                        in_channels=in_channels, 
                        d_model=d_model, 
                        depth=depth, 
                        temporal_depth=temporal_depth,
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias,
                        positional_embedding_dropout=positional_embedding_dropout,
                        attention_dropout=attention_dropout, 
                        projection_dropout=projection_dropout, 
                        dropout_1=dropout_1, 
                        dropout_2=dropout_2, 
                        pre_norm=pre_norm,
                        classification_head=classification_head, 
                        num_classes=num_classes,
                        return_preclassifier=return_preclassifier, 
                        return_prelogits=return_prelogits, 
                        weight_init=weight_init, 
                        weight_load=weight_load, 
                        model_official=model_official,
                        return_intermediate=return_intermediate
                    )
        
        self.caption_decoder = CaptionDecoder(vocab_size=vocab_size, 
                        seq_len=seq_len, 
                        d_model=d_model, 
                        embedding_matrix=embedding_matrix, 
                        emb_weights_req_grad=emb_weights_req_grad, 
                        depth=depth, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        positional_embedding_dropout=positional_embedding_dropout,
                        attention_dropout=attention_dropout, 
                        projection_dropout=projection_dropout, 
                        dropout_1=dropout_1, 
                        dropout_2=dropout_2, 
                        pre_norm=pre_norm,
                        weight_init=weight_init, 
                        weight_load=weight_load, 
                        model_official=model_official, 
                        return_intermediate=False)
        
    # TODO - use log softmax?
    def forward(self, obj):

        """
        Performs a forward pass on the Transformer model
  
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

                                - "pred_captions": All captions in a batch with shape (total_caption_num, seq_len, vocab_size)

                                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                                    dictionaries containing the two above keys for each decoder layer.

            indices (list): matching between the outputs of the last layer and the targets
                            list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))

        """

        x = obj['video_tensor']    # (batch_size, in_channels, num_frames, img_size, img_size)

        # (1, batch_size, num_tokens, d_model) OR (depth, batch_size, num_tokens, d_model)
        res = self.transformer(x, self.query_embedding.weight) 

        # (1, batch_size, num_tokens, num_classes + 1) OR (depth, batch_size, num_tokens, num_classes + 1)
        outputs_class = self.class_embedding(res).softmax(dim=-1)

        # (1, batch_size, num_tokens, 2) OR (depth, batch_size, num_tokens, 2)
        outputs_segment = self.segment_embedding(res).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_segment[-1]}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))
        indices = self.matcher(out, obj['video_target']) 

        # (1, total_caption_num, seq_len, vocab_size) OR (depth, total_caption_num, seq_len, vocab_size)
        outputs_captions = self.caption_decoder(obj['cap_tensor'], memory, positional_embedding)

        out["pred_captions"] = outputs_captions[-1]

        # TODO - indices for aux loss
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment, outputs_captions)

        return out, indices
    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segment, outputs_captions):
        return [{'pred_logits': a, 'pred_segments': b, 'pred_captions': c}
                for a, b, c in zip(outputs_class[:-1], outputs_segment[:-1], outputs_captions[:-1])]

    