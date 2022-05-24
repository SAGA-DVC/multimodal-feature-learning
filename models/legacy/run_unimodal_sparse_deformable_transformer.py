import torch
import torch.nn.functional as F
from torch import nn
from models.unimodal_sparse_deformable_transformer import build_sparse_deformable_transformer
from models.base_encoder import build_base_encoder
from config.config_dvc import load_config
from models.modules.misc_modules import decide_two_stage


cfg = load_config()
base_encoder = build_base_encoder(cfg.dvc.sparse_detr)
transformer = build_sparse_deformable_transformer(cfg.dvc.sparse_detr)


# vf = dt['video_tensor']  # (batch_size, num_token, dmodel) or  (video_num, video_len, video_dim)
# mask = ~ dt['video_mask']  # (batch_size, num_token)
# duration = dt['video_length'][:, 1] # (batch_size)
vf = torch.rand([2, 981, 768])
video_mask = (torch.rand((2, 981))<0.5)
duration = torch.rand([2])


print("start")
# criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,,focal_gamma=args.focal_gamma, opt=args)
criterion = {}
batch_size, L, C = vf.shape
query_embed = nn.Embedding(cfg.dvc.sparse_detr.num_queries, cfg.dvc.sparse_detr.hidden_dim * 2)


#   base encoder for video
video_srcs, video_masks, video_pos = base_encoder(vf, video_mask, duration)



#   forword encoder
video_src_flatten, video_temporal_shapes, video_level_start_index, video_valid_ratios, video_lvl_pos_embed_flatten, video_mask_flatten, backbone_output_proposals, backbone_topk_proposals, sparse_token_nums = transformer.prepare_encoder_inputs(video_srcs, video_masks, video_pos)


video_memory = transformer.forward_encoder(video_src_flatten, video_temporal_shapes, video_level_start_index, video_valid_ratios, video_lvl_pos_embed_flatten, video_mask_flatten, backbone_output_proposals, backbone_topk_proposals, sparse_token_nums)    #   (batch_size, sum of num_token in all level, dmodel) #   Multi-scale frame features

print("output.shape", video_memory[0].shape)
print("sampling_locations_all.shape", video_memory[1].shape)
print("attn_weights_all.shape", video_memory[2].shape)
# print("enc_inter_outputs_class.shape", video_memory[3].shape)
# print("enc_inter_outputs_class.shape", video_memory[4].shape)

# #   decoder
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

query_features, inter_references, sampling_locations_dec, attn_weights_dec = transformer.forward_decoder(tgt, reference_points, video_memory[0], video_temporal_shapes,
                                                        video_level_start_index, video_valid_ratios,  query_embed, 
                                                        video_mask_flatten, proposals_mask, disable_iterative_refine)

print("query_features", query_features.shape)
print("inter_reference", inter_references.shape)
print("sampling_locations_dec", sampling_locations_dec.shape)
print("attn_weights_dec", attn_weights_dec.shape)
#   query_features -> (number of decoder_layers, batch_size, num_queries, dmodel)
#   inter_references = (number of decoder_layers, batch_size, num_queries, 1)

