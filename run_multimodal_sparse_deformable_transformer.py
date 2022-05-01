import torch
import torch.nn.functional as F
from torch import nn
from models.multimodal_sparse_deformable_transformer import build_multimodal_sparse_deforamble_transformer
from models.base_encoder import build_base_encoder
from config.config_dvc import load_config
from models.modules.misc_modules import decide_two_stage


cfg = load_config()
base_encoder = build_base_encoder(cfg.dvc.sparse_detr)
transformer = build_multimodal_sparse_deforamble_transformer(cfg.dvc.sparse_detr)


# vf = dt['video_tensor']  # (batch_size, num_token, dmodel) or  (video_num, video_len, video_dim)
# mask = ~ dt['video_mask']  # (batch_size, num_token)
# duration = dt['video_length'][:, 1] # (batch_size)
vf = torch.rand([2, 981, 768])
video_mask = (torch.rand((2, 981))<0.5)
duration = torch.rand([2])


# af = dt['audio__tensor']  # (batch_size, audio_num_token, dmodel) 
# audio_mask = ~ dt['audio__mask']  # (batch_size, audio_num_token)
# duration = dt['audio_length'][:, 1] # (batch_size)
af =torch.rand([2, 981, 768])
audio_mask = (torch.rand((2, 981))<0.5)


print("start")
# criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,,focal_gamma=args.focal_gamma, opt=args)
criterion = {}
batch_size, L, C = vf.shape
query_embed = nn.Embedding(cfg.sparse_detr.num_queries, cfg.sparse_detr.hidden_dim * 2)


#   base encoder for video
video_srcs, video_masks, video_pos = base_encoder(vf, video_mask, duration)


#   base encoder for audio
audio_srcs, audio_masks, audio_pos = base_encoder(af, audio_mask, duration)



#   forword encoder
# video_input = {src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, backbone_output_proposals, backbone_topk_proposals, sparse_token_nums}
# audio_input = {src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, backbone_output_proposals, backbone_topk_proposals, sparse_token_nums}
video_input = transformer.prepare_encoder_inputs(video_srcs, video_masks, video_pos)
audio_input = transformer.prepare_encoder_inputs(audio_srcs, audio_masks, audio_pos)


video_memory, video_sampling_locations_enc, video_attn_weights_enc, audio_memory, audio_sampling_locations_enc, audio_attn_weights_enc, video_enc_inter_outputs_class, video_enc_inter_outputs_coords, audio_enc_inter_outputs_class, audio_enc_inter_outputs_coords = transformer.forward_encoder(video_input, audio_input)


# print("video_input['valid_ratios']", video_input['valid_ratios'])
# print("audio_input['audio_ratios']", audio_input['valid_ratios'])





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

query_features, inter_references, video_sampling_locations_dec, video_attn_weights_dec, audio_sampling_locations_dec, audio_attn_weights_dec = transformer.forward_decoder(tgt, reference_points, video_memory, video_input, audio_memory, audio_input, query_embed, proposals_mask, disable_iterative_refine)





# print("query_features", query_features.shape)
# print("inter_reference", inter_references.shape)
#   query_features -> (number of decoder_layers, batch_size, num_queries, dmodel)
#   inter_references = (number of decoder_layers, batch_size, num_queries, 1)

