import sys
import os
from pathlib import Path
import random, time, datetime, json
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import wandb

from models import build_model_and_criterion
from config.config_dvc import load_config
from utils.misc import *
from engine import train_one_epoch, evaluate

# from dataset.anet import build_dataset as build_dataset_without_raw_videos, collate_fn as collate_fn_without_raw_videos
from dataset.anet_video import build_dataset as build_dataset_without_raw_videos, collate_fn as collate_fn_without_raw_videos

# from dataset.anet_with_raw_video import build_dataset as build_dataset_with_raw_videos, collate_fn as collate_fn_with_raw_videos
from dataset.anet_with_raw_video_audio import build_dataset as build_dataset_with_raw_videos, collate_fn as collate_fn_with_raw_videos

def main(args):
    init_distributed_mode(args.distributed)

    # wandb logging is in main.py, engine.py and utils.plots.py
    if args.wandb.on and is_main_process():
        wandb.init(project=args.wandb.project, 
                entity=args.wandb.entity, 
                config=args.to_dict(), 
                notes=args.wandb.notes)
        # wandb.run.name = args.wandb.run_name

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if args.use_raw_videos:
        build_dataset = build_dataset_with_raw_videos
        collate_fn = collate_fn_with_raw_videos
    
    # uses encoded video features instead of raw video features
    else:
        build_dataset = build_dataset_without_raw_videos
        collate_fn = collate_fn_without_raw_videos
    
    dataset_train = build_dataset(video_set='train', args=args.dataset.activity_net)
    dataset_val = build_dataset(video_set='val', args=args.dataset.activity_net)

    if args.distributed.is_distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=partial(collate_fn, pad_idx=dataset_train.PAD_IDX, args=args.dataset.activity_net), 
                                   num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, 
                                collate_fn=partial(collate_fn, pad_idx=dataset_train.PAD_IDX, args=args.dataset.activity_net), 
                                num_workers=args.num_workers)

    output_dir = Path(args.output_dir)

    # TODO - pass dataset or specific params?
    model, criterion = build_model_and_criterion(args.dvc, dataset_train, args.use_differentiable_mask)
    model.to(device)
    criterion.to(device)

    print('Model and criterion initialized')

    model_without_ddp = model

    if args.distributed.is_distributed:
        print('Started wrapping model in DDP constructor')

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.distributed.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

        print('Finished wrapping model in DDP constructor')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # for a,b in model.named_parameters():\
    #     print(a, b.shape)
    print(f'number of params: {n_parameters / 1000000} M')

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    gt_json = import_ground_truths_for_eval(args.eval.references)

    if args.model_mode == "testing":
        start_time = time.time()
        if args.distributed.is_distributed:
            sampler_train.set_epoch(args.start_epoch)

        val_stats = evaluate(model, criterion, data_loader_val, dataset_train.vocab, args.print_freq, device, args.start_epoch, args, args.wandb.on, gt_json, val_mode="teacher_forcing")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Total testing time:", total_time_str)
    
    else:
        AssertionError(f'model_mode should be testing. It is {args.model_mode}')
    

if __name__ == '__main__':
    args = load_config()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.submission_dir:
        Path(args.submission_dir).mkdir(parents=True, exist_ok=True)
    main(args)