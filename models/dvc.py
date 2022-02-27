""" 
DVC model
"""

import torch
import torch.nn as nn

from .transformer import Transformer
from .modules import FFN


class DVC(nn.Module):
    def __init__(self, model_name, num_frames_in, img_size=224, spatial_patch_size=16, temporal_patch_size=1,
                tokenization_method='central frame', in_channels=3, d_model=768, depth=12, temporal_depth=4, num_heads=12, 
                mlp_ratio=4., qkv_bias=True, positional_embedding_dropout=0., attention_dropout=0., 
                projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True, classification_head=False, 
                num_classes=None, num_queries=100, aux_loss=False,
                return_preclassifier=True, return_prelogits=False, weight_init=False, weight_load=False, model_official=None,
                return_intermediate=False):
        
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

                                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                                    dictionaries containing the two above keys for each decoder layer.
        """

        x = obj['video_tensor'] # (batch_size, in_channels, num_frames, img_size, img_size)

        # (1, batch_size, num_tokens, d_model) OR (depth, batch_size, num_tokens, d_model)
        res = self.transformer(x, self.query_embedding.weight) 

        # (1, batch_size, num_tokens, num_classes + 1) OR (depth, batch_size, num_tokens, num_classes + 1)
        outputs_class = self.class_embedding(res)

        # (1, batch_size, num_tokens, 2) OR (depth, batch_size, num_tokens, 2)
        outputs_segment = self.segment_embedding(res).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_segment[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment)

        return out
    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segment):
        return [{'pred_logits': a, 'pred_segments': b}
                for a, b in zip(outputs_class[:-1], outputs_segment[:-1])]

    