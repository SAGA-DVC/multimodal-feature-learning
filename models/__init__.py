import sys
# sys.path.insert(0, '..')

import torch
import timm
from .dvc import DVC
from .matcher import build_matcher
from .criterion import SetCriterion
from config.config_dvc import load_config


def build_model_and_criterion(args):

    device = torch.device(args.device)

    model_official = None
    
    if args.model_official is not None:
        model_official = timm.create_model(args.model_official, pretrained=True)
        model_official.eval()


    model = DVC(model_name=args.model_name, 
                num_frames_in=args.num_frames_in, 
                img_size=args.img_size, 
                spatial_patch_size=args.spatial_patch_size, 
                temporal_patch_size=args.temporal_patch_size,
                tokenization_method=args.tokenization_method, 
                in_channels=args.in_channels, 
                d_model=args.d_model, 
                depth=args.depth, 
                temporal_depth=args.temporal_depth,
                num_heads=args.num_heads, 
                mlp_ratio=args.mlp_ratio, 
                qkv_bias=args.qkv_bias,
                positional_embedding_dropout=args.positional_embedding_dropout,
                attention_dropout=args.attention_dropout, 
                projection_dropout=args.projection_dropout, 
                dropout_1=args.dropout_1, 
                dropout_2=args.dropout_2, 
                pre_norm=args.pre_norm,
                classification_head=args.classification_head, 
                num_classes=args.num_classes,
                num_queries=args.num_queries,
                aux_loss=args.aux_loss,
                return_preclassifier=args.return_preclassifier, 
                return_prelogits=args.return_prelogits, 
                weight_init=args.weight_init, 
                weight_load=args.weight_load, 
                model_official=model_official,
                return_intermediate=args.return_intermediate
            )

    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 
                'loss_bbox': args.bbox_loss_coef,
                'loss_giou': args.giou_loss_coef}

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'segments', 'cardinality']

    criterion = SetCriterion(num_classes=args.num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)

    # # postprocessors = {'bbox': PostProcess(args)}

    return model, criterion


# model, criterion = build_model_and_criterion(load_config().dvc)