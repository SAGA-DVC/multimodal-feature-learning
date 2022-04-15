import torch
import torch.nn.functional as F
from torch import nn
from deformable_transformer import build_deforamble_transformer
from base_encoder import build_base_encoder
from config import load_config

from .modules.misc_modules import decide_two_stage


cfg = load_config()
base_encoder = build_base_encoder(cfg.detr)
transformer = build_deforamble_transformer(cfg.detr)


# vf = dt['video_tensor']  # (batch_size, num_token, dmodel) or  (video_num, video_len, video_dim)
# mask = ~ dt['video_mask']  # (batch_size, num_token)
# duration = dt['video_length'][:, 1] # (batch_size)
vf = torch.rand([2, 10, 500])
mask = torch.randint(1,(2, 10))
duration = torch.rand([2])

# criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,,focal_gamma=args.focal_gamma, opt=args)
criterion = {}

N, L, C = vf.shape
# assert N == 1, "batch size must be 1."
query_embed = nn.Embedding(cfg.detr.num_queries, cfg.detr.hidden_dim * 2)


#   base encoder
srcs, masks, pos = base_encoder(vf, mask, duration)
#   print("src shape: ", len(srcs))    #   [[2, 512, 10]  [2, 512, 5] [2, 512, 3] [2, 512, 2]]   list[[batch_size, dmodel, num-token]]
#   print("masks shape: ", len(masks))    #   [[2,10], [2,5], [2,3], [2,2]]
#   print("pos shape: ", len(pos))    #   [[2, 512, 10]  [2, 512, 5] [2, 512, 3] [2, 512, 2]]


#   forword encoder
src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = transformer.prepare_encoder_inputs(srcs, masks, pos)
memory = transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)    #   (batch_size, sum of num_token in all level, dmodel) #   Multi-scale frame features


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
    proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()  #   (batch_size, num_queries)
    init_reference, tgt, reference_points, query_embed = transformer.prepare_decoder_input_query(batch_size, query_embed)

output, inter_references = transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                        level_start_index, valid_ratios, query_embed,
                                                        mask_flatten, proposals_mask, disable_iterative_refine)


#   output -> (number of decoder_layers, batch_size, num_queries, dmodel)
#   inter_reference = (number of decoder_layers, batch_size, num_queries, 1)

