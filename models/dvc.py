""" 
DVC model for event segmentation and captioning
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .vivit import VideoVisionTransformer
from .decoder import Decoder
from .caption_decoder import CaptionDecoder
from .modules import PositionalEmbedding, FFN
from .load_weights import load_positional_embeddings

# TODO - src mask
class DVC(nn.Module):
    def __init__(self, model_name, num_frames_in, img_size=224, spatial_patch_size=16, temporal_patch_size=1,
                tokenization_method='central frame', in_channels=3, d_model=768, 
                vocab_size=1000, seq_len=20, embedding_matrix=None, emb_weights_req_grad=False,
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

        # for encoder
        num_frames = num_frames_in // temporal_patch_size
        num_patches = (img_size // spatial_patch_size) ** 2
        
        self.positional_embedding_layer = None
        self.spatial_positional_embedding_layer = None

        if model_name == 'spatio temporal attention':
            self.positional_embedding_layer = PositionalEmbedding((1, num_frames * num_patches + 1, d_model), positional_embedding_dropout) 
            
        elif model_name == 'factorised encoder':
            self.spatial_positional_embedding_layer = PositionalEmbedding((1, num_patches + 1, d_model), positional_embedding_dropout)
            self.positional_embedding_layer = PositionalEmbedding((1, num_frames + 1, d_model), positional_embedding_dropout)

        else:
            self.positional_embedding_layer = PositionalEmbedding((1, num_frames, num_patches, d_model), positional_embedding_dropout)


        self.vivit = VideoVisionTransformer(model_name=model_name, 
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
                        model_official=model_official
                    )


        self.decoder = Decoder(d_model=d_model, 
                        depth=depth, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias,  
                        attention_dropout=attention_dropout, 
                        projection_dropout=projection_dropout, 
                        dropout_1=dropout_1, 
                        dropout_2=dropout_2, 
                        pre_norm=pre_norm,
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
        

        if weight_load and model_official is not None:
            self.load_weights(model_official)

        elif weight_init:
            self.init_weights()



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

        x = obj['video_tensor']    # (batch_size, in_channels, num_frames, img_size, img_size)
        captions = obj['cap_tensor'][:, :-1]    # (total_caption_num, max_caption_length - 1) - <eos> token should be the last predicted token 
        
        # Encoder
        # (batch_size, num_frames * num_patches + 1, d_model) OR
        # (batch_size, num_frames + 1, d_model) OR 
        # (batch_size, num_frames, num_patches, d_model) 
        feats = self.vivit(x, self.positional_embedding_layer, self.spatial_positional_embedding_layer)

        # TODO - check grad later
        if self.vivit.model_name == 'factorised self attention' or self.vivit.model_name == 'factorised dot product attention':
            feats = feats.reshape(feats.shape[0], -1, feats.shape[-1])


        # Decoder
        query_embedding_weight = self.query_embedding.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)    # (batch_size, num_queries, d_model)
        target = torch.zeros_like(query_embedding_weight)

        # (1, batch_size, num_queries, d_model) OR # (depth, batch_size, num_queries, d_model)
        res = self.decoder(target=target, memory=feats, 
                        positional_embedding_layer=self.positional_embedding_layer, query_embedding=query_embedding_weight, mask=None)


        # (1, batch_size, num_queries, num_classes + 1) OR (depth, batch_size, num_queries, num_classes + 1)
        outputs_class = self.class_embedding(res).softmax(dim=-1)

        # (1, batch_size, num_queries, 2) OR (depth, batch_size, num_queries, 2)
        outputs_segment = self.segment_embedding(res).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_segment[-1]}


        # Caption Decoder
        # Retrieve the matching between the outputs of the last layer and the targets
        # list (len=batch_size) of tuple of tensors (tuple dimensions=(2, gt_target_segments))
        indices = self.matcher(out, obj['video_target']) 

        with torch.no_grad():
            max_gt_target_segments = obj['gt_segments'].shape[1]

            # (nb_target_segments, num_tokens, d_model)
            memory = self.get_segment_features(feats, out['pred_segments'], indices, max_gt_target_segments)
        
        # (1, total_caption_num, max_caption_length - 1, vocab_size) OR (depth, total_caption_num, max_caption_length - 1, vocab_size)
        outputs_captions = self.caption_decoder(captions, memory)

        out["pred_captions"] = outputs_captions[-1]    # (total_caption_num, max_caption_length - 1, vocab_size)

        # TODO - indices for aux loss
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment, outputs_captions)

        return out, indices
    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segment, outputs_captions):
        return [{'pred_logits': a, 'pred_segments': b, 'pred_captions': c}
                for a, b, c in zip(outputs_class[:-1], outputs_segment[:-1], outputs_captions[:-1])]


    # TODO - make more efficient
    @torch.no_grad
    def get_segment_features(self, features, pred_segments, indices, max_gt_target_segments):

        batch_size, num_tokens, d_model = features.shape

        pred_segment_boundaries = torch.zeros(batch_size, max_gt_target_segments, 2)

        for i, pred_idx, _ in enumerate(indices):
            pred_segment_boundaries[i, :pred_idx.shape[0]] = pred_segments[i, pred_idx.long()]
        
        # (batch_size, max_gt_target_segments, num_tokens, d_model) AND (batch_size, max_gt_target_segments)
        pred_features, pred_features_mask = self.crop_segments(features, pred_segment_boundaries, indices, max_gt_target_segments)
        
        pred_features = pred_features.reshape(-1, num_tokens, d_model)
        pred_features_mask = pred_features_mask.reshape(-1)
        
        # removes extra captions(padding) added to satisfy dimension constraints of tensors
        pred_features = pred_features[pred_features_mask == True]    # (nb_target_segments, num_tokens, d_model)
        return pred_features    


    # TODO - padding like in BMT??
    @torch.no_grad
    def crop_segments(self, features, pred_segment_boundaries, indices, max_gt_target_segments):
        
        batch_size, num_tokens, d_model = features.shape
        
        start_quantile = pred_segment_boundaries[:, :, 0]    # (batch_size, max_gt_target_segments)
        end_quantile = pred_segment_boundaries[:, :, 1]    # (batch_size, max_gt_target_segments)
        start_idx = (num_tokens * start_quantile).long()    # (batch_size, max_gt_target_segments)
        end_idx = (num_tokens * end_quantile).long    # (batch_size, max_gt_target_segments)

        if start_idx >= end_idx:
            if start_idx >= num_tokens:
                start_idx = num_tokens - 1

            elif end_idx >= num_tokens:
                end_idx = num_tokens + 1
            
            else:
                end_idx = start_idx + 1
            
        pred_features = torch.zeros(batch_size, max_gt_target_segments, num_tokens, d_model)
        pred_features_mask = torch.zeros(batch_size, max_gt_target_segments, dtype=torch.bool)

        for i in range(batch_size):
            gt_target_segments = len(indices[i][0])
            for j in range(gt_target_segments):
                pred_features[i, j] = features[i, start_idx[i, j]:end_idx[i, j], :]
                pred_features_mask[i, j] = True

        return pred_features, pred_features_mask


    
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