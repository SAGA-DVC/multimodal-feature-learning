# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import time, datetime
import os
import json
from pathlib import Path
import sys
from typing import Iterable
from collections import defaultdict
import wandb
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from utils.misc import MetricLogger, SmoothedValue, reduce_dict, is_main_process
from utils.preds_postprocess import get_sample_submission, get_src_permutation_idx, denormalize_segments, pprint_eval_scores, save_submission
from utils.plots import plot_grad_flow_line_plot, plot_grad_flow_bar_plot
from evaluation_action_recog.get_detection_performance import run_eval
from models.matcher import build_matcher


def train_one_epoch(model, criterion, data_loader, optimizer, print_freq, device, epoch, args, wandb_log, inverted_action_labels_dict):
    
    """
    Trains the given model for 1 epoch and logs various metrics such as model losses and those associated with the training loop.

    Parameters:
        `model` (torch.nn.Module) : Model to be trained
        `criterion` (torch.nn.Module) : Losses used to train the model
        `data_loader` (Iterable) : DataLoader for the associated dataset (ActivityNet)
        `optimizer` (torch.optim.Optimizer) : Optimizer fused to train the model
        `device` (torch.device) : the device on which the data has to be placed. It should be the same device that given model resides on.
        `epoch` (int) : Epoch number
        `args` (ml_collections.ConfigDict) : config parameters
        `wandb_log` (boolean) : If True, log metrics in wandb
    
    Returns: dictionary with keys as all the losses calculated by the criterion and values as their corresponding global average across all devices.
    """

    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="\t")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = args.print_freq

    def map_id_to_classname(cid):
        return inverted_action_labels_dict[str(cid)]

    for (batch_idx, obj) in enumerate(metric_logger.log_every(data_loader, print_freq, wandb_log, header)):

        obj = {key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in obj.items()}
        obj['video_target'] = [{key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in vid_info.items()} 
                                for vid_info in obj['video_target']]

        obj = defaultdict(lambda: None, obj)

        outputs = model(obj)
        
        loss_dict = criterion(outputs, obj)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        # print("Logits, segments:", outputs['pred_logits'].shape, outputs['pred_segments'].shape)
        # classes_id = torch.argmax(outputs['pred_logits'], dim=-1)
        # mapper = np.vectorize(map_id_to_classname)
        # classes_names = mapper(classes_id.cpu().detach().numpy()).reshape(-1)

        # if batch_idx % 100 == 0:
        #     # plot_grad_flow_line_plot(model.named_parameters(), epoch, batch_idx, args.output_dir, wandb_log)
        #     plot_grad_flow_bar_plot(model.named_parameters(), epoch, batch_idx, args.output_dir, wandb_log)

        #     train_classes_path = Path(os.path.join(args.submission_dir, 'train'))
        #     if not os.path.exists(train_classes_path):
        #         train_classes_path.mkdir(parents=True, exist_ok=True)

        #     # src_classes_string = classes_to_string(obj['cap_tensor'])
        #     # tgt_classes_string = classes_to_string(classes)    # (total_caption_num, max_caption_length - 1)

            
        #     res = {}
        #     for src, tgt in zip(src_classes_string, tgt_classes_string):
        #         res[src] = tgt

        #     if args.output_dir and is_main_process():
        #         with (train_classes_path / "train_classes.json").open("a") as f:
        #             json.dump(res, f, indent=4)
                
        #         if args.wandb.on:
        #             wandb.save(os.path.join(train_classes_path, "train_classes.json"))

        if args.clip_max_norm > 0:
            clip_grad_norm_(model.parameters(), args.clip_max_norm)
        
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if wandb_log and is_main_process():
            substring_list = [f'_{i}' for i in range(12)]
            wandb_log_metrics(
                phase="train",
                loss=loss_value,
                loss_dict=loss_dict_reduced_scaled,
                epoch=epoch,
                batch_idx=batch_idx,
                substring_list=substring_list
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"\nAveraged train stats for epoch [{epoch}]: ", metric_logger, "\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# TODO: wandb scores (combine scores across batches)
@torch.no_grad()
def evaluate(model, criterion, data_loader, print_freq, device, epoch, args, wandb_log, inverted_action_labels_dict, gt_val_json):
    
    """
    Inference on given data and save the results.

    Parameters:
        `model` (torch.nn.Module) : Trained Model
        `criterion` (torch.nn.Module) : Losses used to train the model
        `data_loader` (Iterable) : DataLoader for the test dataset (ActivityNet)
        `vocab` (torchtext.vocab.Vocab): mapping of all the words in the training dataset to indices and vice versa)
        `device` (torch.device) : the device on which the data has to be placed. It should be the same device that given model resides on.
        `eval_args` (ml_collections.ConfigDict) : config params for run_eval
    
    Returns: ???
    """

    model.eval()
    criterion.eval()

    submission_json_epoch = get_sample_submission()

    metric_logger = MetricLogger(delimiter="\t")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = args.print_freq

    def map_id_to_classname(cid):
        return inverted_action_labels_dict[str(cid)]

    for (batch_idx, obj) in enumerate(metric_logger.log_every(data_loader, print_freq, wandb_log, header)):
        
        submission_json_batch = get_sample_submission()

        obj = {key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in obj.items()}
        obj['video_target'] = [{key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in vid_info.items()} 
                                for vid_info in obj['video_target']]

        obj = defaultdict(lambda: None, obj)

        outputs = model(obj)
        
        loss_dict = criterion(outputs, obj)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        # EVALUATION SCORES
        # segments
        # TODO indices, segment_batch_id??
        # idx = get_src_permutation_idx(indices)
        
        segment_batch_id = torch.Tensor([[i for j in range(outputs['pred_logits'].shape[1])] for i in range(outputs['pred_logits'].shape[0])]).long()
        # print(segment_batch_id.shape, segment_batch_id[0])

        # classes
        # classes_id = torch.argmax(outputs['pred_logits'][..., :-1], dim=-1)    # (batch_size, num_queries, num_classes)

        # (batch_size, num_queries), (batch_size, num_queries)
        scores, classes_id = torch.max(outputs['pred_logits'][..., :-1], dim=-1)
        scores = scores.cpu().detach()

        keep = scores > 0.01   #  (batch_size, num_queries)
        classes_id_keep = classes_id[keep]    # (batch_size * <num_queries)
        scores_keep = scores[keep]    # (batch_size * <num_queries)
        segment_batch_id_keep = segment_batch_id[keep]    # (batch_size * <num_queries)

        mapper = np.vectorize(map_id_to_classname)
        classes_names = mapper(classes_id_keep.cpu().detach().numpy()).reshape(-1)    # (batch_size * <num_queries)

        video_durations = list(obj['video_length'][:, 1])
        denormalized_segments = denormalize_segments(outputs['pred_segments'][keep].reshape(-1, 2), video_durations, segment_batch_id_keep)
        # print("Video_DUR: ",video_durations, outputs['pred_segments'].shape, denormalized_segments.shape)

        for i, batch_id in enumerate(segment_batch_id_keep):
            video_id = obj['video_key'][batch_id]
            append_result_to_json_submission_file(video_id, submission_json_batch, classes_names[i], denormalized_segments[i], scores_keep[i])
            append_result_to_json_submission_file(video_id, submission_json_epoch, classes_names[i], denormalized_segments[i], scores_keep[i])

        avg_mAP = run_eval(args.eval, gt_val_json, submission_json_batch)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(**avg_mAP)

        if wandb_log and is_main_process():
            loss_dict_reduced_scaled.update(avg_mAP)
            substring_list = [f'_{i}' for i in range(12)]
            wandb_log_metrics(
                phase="val",
                loss=loss_value,
                loss_dict=loss_dict_reduced_scaled,
                epoch=epoch,
                batch_idx=batch_idx,
                substring_list=substring_list
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"\nAveraged val stats for epoch [{epoch}]: ", metric_logger, "\n")

    # TODO - check if run_eval can be removed and we can instead avg scores in above loop
    # scores = run_eval(args.eval, submission_json_epoch)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update(avg_mAP)

    val_caption_path = Path(os.path.join(args.submission_dir, 'val'))
    if not os.path.exists(val_caption_path):
        val_caption_path.mkdir(parents=True, exist_ok=True)
    
    if args.output_dir and is_main_process():
        if args.save_submission:
            save_submission(submission_json_epoch, os.path.join(val_caption_path, f"E{epoch}_submission.json"))
        
        if wandb_log:
            wandb.save(os.path.join(val_caption_path, f"E{epoch}_submission.json"))

    return return_dict



# TODO - no grad reqd??
@torch.no_grad()
def wandb_log_metrics(phase, loss, loss_dict, epoch, batch_idx, substring_list):
    log = {
        "epoch": epoch,
        "batch": batch_idx,
        "loss": loss,
    }
    for key, value in loss_dict.items():
        if all(substring not in key for substring in substring_list) or 'Bleu' in key:    # don't log aux loss in charts
            if isinstance(value, float):
                log[key] = value
            else:
                log[key] = value.item()

    log_dict = {f"{phase}-{key}": value for key, value in log.items()}
    # print(log_dict)
    wandb.log(log_dict)


def append_result_to_json_submission_file(video_id, submission_json_batch, class_name, denormalized_segments, score):
    if video_id not in submission_json_batch['results']:
        submission_json_batch['results'][video_id] = []

    submission_json_batch['results'][video_id].append({
        'segment': [denormalized_segments[0].item(), denormalized_segments[1].item()],
        'label': class_name,
        'score': score.item()
    })