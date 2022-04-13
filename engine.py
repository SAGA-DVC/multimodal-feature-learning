# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from collections import defaultdict

import torch
from torch.nn.utils import clip_grad_norm_
from utils.misc import MetricLogger, SmoothedValue, reduce_dict
from utils.preds_postprocess import get_sample_submission, get_src_permutation_idx, denormalize_segments, captions_to_string, pprint_eval_scores, save_submission
from evaluation.evaluate import run_eval

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_grad_flow_v1(named_parameters, epoch, batch_idx):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.grad == None:
            print(n, p.requires_grad)
            print("Grad None!!")
        if(p.requires_grad) and ("bias" not in n) and (p.grad != None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=5)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.005) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('output/grads/E{}_B{}_line.png'.format(epoch, batch_idx), bbox_inches='tight')


def plot_grad_flow_v2(named_parameters, epoch, batch_idx):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if p.grad == None:
            print(n, p.requires_grad)
            print("Grad None!!")
        if(p.requires_grad) and ("bias" not in n) and (p.grad != None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=5)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.005) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('output/grads/E{}_B{}_bar.png'.format(epoch, batch_idx), bbox_inches='tight')


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, args, wandb_log, wandb):
    
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
    
    Returns: dictionary with keys as all the losses calculated by the criterion and values as their corresponding global average across all devices.
    """

    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="\t")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 1

    for (batch_idx, obj) in enumerate(metric_logger.log_every(data_loader, print_freq, wandb_log, wandb, header)):

        obj = {key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in obj.items()}
        obj['video_target'] = [{key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in vid_info.items()} 
                                for vid_info in obj['video_target']]

        obj = defaultdict(lambda: None, obj)
        outputs, indices = model(obj)
        
        loss_dict = criterion(outputs, obj, indices)
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

        # plot_grad_flow_v1(model.named_parameters(), epoch, batch_idx)
        # plot_grad_flow_v2(model.named_parameters())

        if args.clip_max_norm > 0:
            clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"\nAveraged stats for epoch [{epoch}]: ", metric_logger, "\n")

    if wandb_log:
        wandb.log({f"Averaged stats for epoch [{epoch}]": str(metric_logger)})

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# TODO: Pass json instead of creating file and passing file path
# TODO: wandb scores (combine scores across batches)
@torch.no_grad()
def evaluate(model, criterion, data_loader, vocab, device, eval_args):
    
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

    submission_json = get_sample_submission()

    for i, obj in enumerate(data_loader):

        obj = {key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in obj.items()}
        obj['video_target'] = [{key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in vid_info.items()} 
                                for vid_info in obj['video_target']]

        obj = defaultdict(lambda: None, obj)

        pred_segments, pred_captions, pred_logits, indices = model(obj, is_training=False, faster_eval=False)
        # print("Pred Shapes Eval: ", pred_segments.shape, pred_captions.shape, pred_logits.shape, indices)

        # EVALUATION SCORES
        # segments
        idx = get_src_permutation_idx(indices)
        # print("IDX: ", idx)

        video_durations = list(obj['video_length'][:, 1])
        denormalized_segments = denormalize_segments(pred_segments[idx], video_durations, idx[0])
        # print("Video_DUR: ",video_durations, pred_segments[idx], denormalized_segments, denormalized_segments.shape)

        # captions
        captions_string = captions_to_string(pred_captions, vocab)
        # print("Captions: ", pred_captions[0], captions_string[0])

        for i, batch_id in enumerate(idx[0]):
            video_id = obj['video_key'][batch_id]
            
            if video_id not in submission_json['results']:
                submission_json['results'][video_id] = []

            submission_json['results'][video_id].append({
                'sentence': captions_string[i],
                'timestamp': [denormalized_segments[i][0].item(), denormalized_segments[i][1].item()]
            })

    save_submission(submission_json, eval_args.submission)
    
    scores = run_eval(eval_args)
    pprint_eval_scores(scores)