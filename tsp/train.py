import datetime
from itertools import chain
import os
import time

import json
import torch
import torchvision

from tsp.model import TSPModel
from tsp.untrimmed_video_dataset import UntrimmedVideoDataset
from tsp.lr_scheduler import WarmupMultiStepLR
from tsp.config import load_config

def epoch_loop(model: TSPModel, criterion, optimizer, lr_scheduler, dataloader, device, epoch, print_freq, label_columns, loss_alphas):
    model.train()
    
    for sample in dataloader:
        clip = sample['clip'].to(device)
        gvf = sample['gvf'].to(device) if 'gvf' in sample else None
        targets = [sample[x].to(device) for x in label_columns]  # [(B, 1), (B, 1)]
        
        outputs = model(clip, gvf=gvf)  # [(B, 1), (B, 1)]

        # compute losses
        head_losses, loss = [], 0
        for output, target, alpha in zip(outputs, targets, loss_alphas):
            head_loss = criterion(output, target)
            head_losses.append(head_loss)
            loss += alpha * head_loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: compute accuracies and log metrics

        lr_scheduler.step()

def evaluate(model: TSPModel, criterion, dataloader, device, epoch, print_freq, label_columns, loss_alphas, output_dir):
    model.eval()
    
    with torch.no_grad():
        for sample in dataloader:
            clip = sample['clip'].to(device, non_blocking=True)
            gvf = sample['gvf'].to(device, non_blocking=True) if 'gvf' in sample else None
            targets = [sample[x].to(device, non_blocking=True) for x in label_columns]
            
            outputs = model(clip, gvf=gvf)

            # compute loss
            head_losses, loss = [], 0
            for output, target, alpha in zip(outputs, targets, loss_alphas):
                head_loss = criterion(output, target)
                head_losses.append(head_loss)
                loss += alpha * head_loss
            
            # TODO: Log metrics
        
        # TODO: Save results to file, print



def main(args):
    cfg = load_config()

    # TODO: Distributed mode setup

    device = torch.device(cfg.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_dir = os.path.join(cfg.data_dir, cfg.train_subdir)
    valid_dir = os.path.join(cfg.data_dir, cfg.valid_subdir)

    print("Loading data")
    label_mappings = []
    for label_mapping_json in cfg.dataset.label_mapping_jsons:
        with open(label_mapping_json) as f:
            label_mapping = json.load(f)
            label_mappings.append(dict(zip(label_mapping, range(len(label_mapping)))))

    float_zero_to_one = torchvision.transforms.Lambda(lambda video: video.permute(3, 0, 1, 2).to(torch.float32) / 255)

    normalize = torchvision.transforms.Normalize(
        mean=[0.43216, 0.394666, 0.37645], 
        std=[0.22803, 0.22145, 0.216989])

    resize = torchvision.transforms.Resize((128, 171))
    
    train_transform = torchvision.transforms.Compose([
        float_zero_to_one,
        resize,
        torchvision.transforms.RandomHorizontalFlip(),
        normalize,
        torchvision.transforms.RandomCrop((112, 112))
    ])

    train_dataset = UntrimmedVideoDataset(
        csv_filename=cfg.dataset.train_csv_filename,
        root_dir=train_dir,
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        clips_per_segment=cfg.video.clip_per_segment,
        temporal_jittering=True,
        transforms=train_transform,
        label_columns=cfg.dataset.label_columns,
        label_mappings=label_mappings,
        global_video_features=cfg.tsp.global_video_features,
        debug=cfg.debug
    )

    valid_transform = torchvision.transforms.Compose([
        float_zero_to_one,
        resize,
        normalize,
        torchvision.transforms.CenterCrop((112, 112))
    ])

    valid_dataset = UntrimmedVideoDataset(
        csv_filename=cfg.dataset.valid_csv_filename,
        root_dir=valid_dir,
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        clips_per_segment=cfg.video.clip_per_segment,
        temporal_jittering=False,
        transforms=valid_transform,
        label_columns=cfg.dataset.label_columns,
        label_mappings=label_mappings,
        global_video_features=cfg.tsp.global_video_features,
        debug=cfg.debug
    )

    print("Creating dataloaders")
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True) if cfg.distributed else None
    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, shuffle=False) if cfg.distributed else None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    print("Creating model")
    tsp_model = TSPModel(
        backbone=cfg.tsp.backbone, 
        num_classes=[len(l) for l in label_mappings],
        num_heads=len(cfg.dataset.label_columns),
        concat_gvf=cfg.tsp.global_video_features is not None,
        **cfg.vivit
    )
    
    tsp_model.to(device)
    if cfg.distributed and cfg.distributed.sync_bn:
        tsp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tsp_model)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # label == -1 => missing label

    if len(cfg.dataset.label_columns) == 1:
        fc_params = tsp_model.action_class_fc.parameters()
    elif len(cfg.label_columns) == 2:
        fc_params = chain(tsp_model.action_class_fc.parameters(), tsp_model.region_fc.parameters())
    else:
        raise NotImplementedError
    
    params = [
        {
            "params": tsp_model.feature_extractor.parameters(),
            "lr": cfg.tsp.backbone_lr * (cfg.distributed.world_size if cfg.distributed else 1),
            "name": "backbone"
        },
        {
            "params": fc_params,
            "lr": cfg.tsp.fc_lr * (cfg.distributed.world_size if cfg.distributed else 1),
            "name": "fc"
        }
    ]

    optimizer = torch.optim.SGD(
        params,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )

    # Scheduler per iteration, not per epoch for warmup that lasts
    warmup_iters = cfg.lr_warmup_epochs * len(train_dataloader)
    lr_milestones = [len(train_dataloader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=cfg.lr_gamma,
        warmup_iters=warmup_iters,
        warmup_factor=cfg.lr_warmup_factor
    )

    model_without_ddp = tsp_model
    if cfg.distributed:
        tsp_model = torch.nn.parallel.DistributedDataParallel(tsp_model, device_ids=[cfg.gpu])
        model_without_ddp = tsp_model.module

    if cfg.resume:
        print(f'Resuming from checkpoint {cfg.resume}')
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint['epoch'] + 1

    if cfg.valid_only:
        epoch = cfg.start_epoch - 1 if cfg.resume else cfg.start_epoch
        evaluate(
            model=tsp_model, 
            criterion=criterion,
            dataloader=valid_dataloader,
            device=device,
            epoch=epoch,
            print_freq=cfg.print_freq,
            label_columns=cfg.dataset.label_columns,
            loss_alphas=cfg.tsp.loss_alphas,
            output_dir=cfg.output_dir
        )
        return
    
    print("Starting training")
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
        
        epoch_loop(
            model=tsp_model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=train_dataloader,
            device=device,
            epoch=epoch,
            print_freq=cfg.print_freq,
            label_columns=cfg.dataset.label_columns,
            loss_alphas=cfg.tsp.loss_alphas
        )

        if cfg.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'cfg': cfg
            }

            # TODO: save checkpoint on master

        if cfg.train_only_one_epoch:
            break
        else:
            evaluate(
                model=tsp_model,
                criterion=criterion,
                dataloader=valid_dataloader,
                device=device,
                epoch=epoch,
                print_freq=cfg.print_freq,
                label_columns=cfg.dataset.label_columns,
                loss_alphas=cfg.tsp.loss_alphas,
                output_dir=cfg.output_dir
            )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    main()

