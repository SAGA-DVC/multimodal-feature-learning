""" 
DETR model
"""

import torch
import torch.nn as nn

from transformer import Transformer
from modules import FFN


class DETR(nn.Module):
    def __init__(self, model_name, num_frames, num_patches, img_size=224, spatial_patch_size=16, temporal_patch_size=1,
                tokenization_method='central frame', in_channels=3, d_model=768, depth=12, temporal_depth=4, num_heads=12, 
                mlp_ratio=4., qkv_bias=True, positional_embedding_dropout=0., attention_dropout=0., 
                projection_dropout=0., dropout_1=0., dropout_2=0., pre_norm=True, classification_head=False, 
                num_classes=None, num_queries=100, aux_loss=False,
                return_preclassifier=True, return_prelogits=False, weight_init=False, weight_load=False, model_official=None,
                return_intermediate=False):
        
        """
        Detr type model
        """

        super(DETR, self).__init__()
        
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        self.query_embedding = nn.Embedding(num_queries, d_model)
        self.class_embedding = nn.Linear(d_model, num_classes + 1)
        self.event_segment_embedding = FFN(d_model, d_model, 3, 3)

        self.transformer = Transformer(model_name=model_name, 
                        num_frames=num_frames, 
                        num_patches=num_patches, 
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
                        return_intermediate=False
                    )
        
    
    def forward(self, x):

        """
        Performs a forward pass on the Transformer model
  
        Parameters:
            x (tensor): Tensor of dimension (batch_size, in_channels, num_frames, img_size, img_size)
        
        Returns:
            out (dictionary) : It returns a dict with the following elements:
                                - "pred_logits": the classification logits (including no-object) for all queries
                                                    shape (batch_size, num_queries, num_classes + 1)
                                - "pred_event_segments": The normalized event segments for all queries, represented as
                                                (center_offset, length, confidence). 
???????
values are normalized in [0, 1]
relative to the size of each individual image (disregarding possible padding).
See PostProcess for information on how to retrieve the unnormalized bounding box.

                                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                                    dictionnaries containing the two above keys for each decoder layer.
        """

        # (1, batch_size, num_tokens, d_model) OR (depth, batch_size, num_tokens, d_model)
        res = self.transformer(x, self.query_embedding.weight) 

        outputs_class = self.class_embedding(res)
        outputs_segment = self.event_segment_embedding(res).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_event_segments': outputs_segment[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment)

        return out
    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segment):
        return [{'pred_logits': a, 'pred_event_segments': b}
                for a, b in zip(outputs_class[:-1], outputs_segment[:-1])]

    