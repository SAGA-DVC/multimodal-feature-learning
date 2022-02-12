import torch
from dvc import DVC
from matcher import build_matcher
from criterion import SetCriterion

def build_model_and_criterion(args):
    
    device = torch.device(args.device)

    model = DVC(**args)

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

    criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    # postprocessors = {'bbox': PostProcess(args)}

    return model, criterion