import torch
import torch.nn.functional as F
from torch import nn
from multimodal_deformable_transformer import build_multimodal_deformable_transformer
from base_encoder import build_base_encoder
from config import load_config

from .modules.misc_modules import decide_two_stage


cfg = load_config()
base_encoder = build_base_encoder(cfg.detr)
transformer = build_multimodal_deformable_transformer(cfg.detr)


# vf = dt['video_tensor']  # (batch_size, num_token, dmodel) or  (video_num, video_len, video_dim)
# mask = ~ dt['video_mask']  # (batch_size, num_token)
# duration = dt['video_length'][:, 1] # (batch_size)
vf = torch.rand([2, 10, 500])
video_mask = torch.randint(1,(2, 10))
duration = torch.rand([2])


# af = dt['audio__tensor']  # (batch_size, audio_num_token, dmodel) 
# audio_mask = ~ dt['audio__mask']  # (batch_size, audio_num_token)
# duration = dt['audio_length'][:, 1] # (batch_size)
af = torch.rand([2, 10, 500])
audio_mask = torch.randint(1,(2, 10))
# duration = torch.rand([2])

print("start")
# criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,,focal_gamma=args.focal_gamma, opt=args)
criterion = {}
batch_size, L, C = vf.shape
query_embed = nn.Embedding(cfg.detr.num_queries, cfg.detr.hidden_dim * 2)


#   base encoder for video
video_srcs, video_masks, video_pos = base_encoder(vf, video_mask, duration)
#   print("src shape: ", len(video_srcs))    #   [[2, 512, 10]  [2, 512, 5] [2, 512, 3] [2, 512, 2]]   list[[batch_size, dmodel, num-token]]
#   print("masks shape: ", len(masks))    #   [[2,10], [2,5], [2,3], [2,2]]
#   print("pos shape: ", len(pos))    #   [[2, 512, 10]  [2, 512, 5] [2, 512, 3] [2, 512, 2]]



#   base encoder for audio
audio_srcs, audio_masks, audio_pos = base_encoder(af, audio_mask, duration)


#   forword encoder
video_src_flatten, video_temporal_shapes, video_level_start_index, video_valid_ratios, video_lvl_pos_embed_flatten, video_mask_flatten = transformer.prepare_encoder_inputs(video_srcs, video_masks, video_pos)

audio_src_flatten, audio_temporal_shapes, audio_level_start_index, audio_valid_ratios, audio_lvl_pos_embed_flatten, audio_mask_flatten = transformer.prepare_encoder_inputs(audio_srcs, audio_masks, audio_pos)

video_memory, audio_memory = transformer.forward_encoder(video_src_flatten, video_temporal_shapes, video_level_start_index, video_valid_ratios, video_lvl_pos_embed_flatten, video_mask_flatten, audio_src_flatten, audio_temporal_shapes, audio_level_start_index, audio_valid_ratios, audio_lvl_pos_embed_flatten, audio_mask_flatten)    #   (batch_size, sum of num_token in all level, dmodel) #   Multi-scale frame features


print("video_memory.shape", video_memory.shape)
print("audio_memory.shape", audio_memory.shape)

#   decoder
transformer_input_type = "queries"
gt_boxes = torch.rand([2, 3, 2])
gt_boxes_mask = torch.randint(1,(2, 3))

two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type,
                                                                                        gt_boxes, gt_boxes_mask, criterion)

if two_stage:
    init_reference, tgt, reference_points, query_embed = transformer.prepare_decoder_input_proposal(proposals)
else:
    query_embed = query_embed.weight
    proposals_mask = torch.ones(batch_size, query_embed.shape[0], device=query_embed.device).bool()  #   (batch_size, num_queries)
    init_reference, tgt, reference_points, query_embed = transformer.prepare_decoder_input_query(batch_size, query_embed)

query_features, inter_references = transformer.forward_decoder(tgt, reference_points, query_embed, proposals_mask, video_memory, video_temporal_shapes,
                                                        video_level_start_index, video_valid_ratios,
                                                        video_mask_flatten, audio_memory, audio_temporal_shapes,
                                                        audio_level_start_index, audio_valid_ratios,
                                                        audio_mask_flatten,  disable_iterative_refine)

print("query_features", query_features.shape)
print("inter_reference", inter_references.shape)
#   query_features -> (number of decoder_layers, batch_size, num_queries, dmodel)
#   inter_references = (number of decoder_layers, batch_size, num_queries, 1)

