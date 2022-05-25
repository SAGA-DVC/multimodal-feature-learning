from pathlib import Path
import pickle

import torch
import numpy as np
import timm
from .deformable.unimodal_deformable_dvc import UnimodalDeformableDVC
from .deformable.multimodal_deformable_dvc import MultimodalDeformableDVC
from .sparse.unimodal_sparse_dvc import UnimodalSparseDVC
from .sparse.multimodal_sparse_dvc import MultimodalSparseDVC
from .regular.dvc import DVC
from .matcher import build_matcher
from .criterion import SetCriterion
from config.config_dvc import load_config

# TODO - file.close()?
def build_model_and_criterion(args, dataset, use_differentiable_mask=False):

    # device = torch.device(args.device)
    
    model_official = None

    matcher = build_matcher(args.matcher)

    if args.use_deformable_detr:
        if len(args.input_modalities) == 1:
            model = UnimodalDeformableDVC(input_modalities=args.input_modalities,
                        num_queries=args.num_queries,
                        d_model=args.d_model, 
                        num_classes=args.num_classes,
                        aux_loss=args.aux_loss,
                        max_eseq_length=args.max_eseq_length,
                        detr_args=args.detr
                    )
        else:
            model = MultimodalDeformableDVC(input_modalities=args.input_modalities,
                        num_queries=args.num_queries,
                        d_model=args.d_model, 
                        num_classes=args.num_classes,
                        aux_loss=args.aux_loss,
                        max_eseq_length=args.max_eseq_length,
                        detr_args=args.detr
                    )

    elif args.use_sparse_detr:
        if len(args.input_modalities) == 1:
            model = UnimodalSparseDVC(input_modalities=args.input_modalities,
                        num_queries=args.num_queries,
                        d_model=args.d_model, 
                        num_classes=args.num_classes,
                        aux_loss=args.aux_loss,
                        max_eseq_length=args.max_eseq_length,
                        sparse_detr_args=args.sparse_detr
                    )
        else:
            model = MultimodalSparseDVC(input_modalities=args.input_modalities,
                        num_queries=args.num_queries,
                        d_model=args.d_model, 
                        num_classes=args.num_classes,
                        aux_loss=args.aux_loss,
                        max_eseq_length=args.max_eseq_length,
                        detr_args=args.detr
                    )

    else :
        model = DVC(input_modalities=args.input_modalities,
                    num_queries=args.num_queries,
                    d_model=args.d_model, 
                    num_classes=args.num_classes,
                    aux_loss=args.aux_loss,
                    max_eseq_length=args.max_eseq_length, 
                    encoder_args=args.encoder,
                    decoder_args=args.decoder
                )

    weight_dict = {'loss_ce': args.cls_loss_coef,
                'loss_counter': args.counter_loss_coef, 
                'loss_bbox': args.bbox_loss_coef,
                'loss_giou': args.giou_loss_coef,
                'loss_self_iou': args.self_iou_loss_coef,
                'loss_mask_prediction': args.mask_prediction_coef,
                'loss_corr': args.corr_coef,
                }

    if use_differentiable_mask:
        weight_dict['loss_context'] = args.context_loss_coef

    # TODO this is a hack
    if args.use_sparse_detr:
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.sparse_detr.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != 'loss_caption'})
        
        if args.sparse_detr.use_enc_aux_loss:
            enc_aux_weight_dict = {}
            for i in range(args.sparse_detr.enc_layers - 1):
                enc_aux_weight_dict.update({k + f'_enc_{i}': v for k, v in weight_dict.items()})

        if args.aux_loss:
            weight_dict.update(aux_weight_dict)
        
        if args.sparse_detr.use_enc_aux_loss:
            weight_dict.update(enc_aux_weight_dict)
    
    elif args.use_deformable_detr:
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.detr.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)


    criterion = SetCriterion(len(args.input_modalities) == 2, num_classes=args.num_classes, matcher=matcher, weight_dict=weight_dict,
                            eos_coef=args.eos_coef, losses=args.losses, smoothing=args.smoothing,
                            focal_alpha=0.25, focal_gamma=2, lloss_gau_mask=args.lloss_gau_mask, lloss_beta=args.lloss_gau_mask)

    return model, criterion