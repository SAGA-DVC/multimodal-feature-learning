import datetime
from itertools import chain
import os
import time
import json
from pprint import pprint

import torch
import torchvision
import wandb
import timm

from models.vivit import VideoVisionTransformer
from tsp.model import TSPModel
from tsp.untrimmed_video_dataset import UntrimmedVideoDataset
from tsp.lr_scheduler import WarmupMultiStepLR
from tsp.config import load_config
from tsp import utils

def epoch_loop(model: TSPModel, criterion, optimizer, lr_scheduler, dataloader, device, epoch, print_freq, label_columns, loss_alphas):
    model.train()
    
    for (batch_idx, batch) in enumerate(dataloader):
        clip = batch['clip'].to(device)
        gvf = batch['gvf'].to(device) if 'gvf' in batch else None
        targets = [batch[x].to(device) for x in label_columns]  # [(B, 2), (B, c)] (len=2)
        outputs = model(clip, gvf=gvf)  # [(B, 2), (B, c)]

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

        compute_and_log_metrics(
            phase="train",
            loss=loss, 
            outputs=outputs, 
            targets=targets, 
            head_losses=head_losses, 
            label_columns=label_columns, 
            epoch=epoch, 
            batch_idx=batch_idx
        )

        lr_scheduler.step()

def evaluate(model: TSPModel, criterion, dataloader, device, epoch, print_freq, label_columns, loss_alphas, output_dir):
    model.eval()
    
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(dataloader):
            clip = batch['clip'].to(device, non_blocking=True)
            gvf = batch['gvf'].to(device, non_blocking=True) if 'gvf' in batch else None
            targets = [batch[x].to(device, non_blocking=True) for x in label_columns]

            outputs = model(clip, gvf=gvf)

            # compute loss
            head_losses, loss = [], 0
            for output, target, alpha in zip(outputs, targets, loss_alphas):
                head_loss = criterion(output, target)
                head_losses.append(head_loss)
                loss += alpha * head_loss
            
            compute_and_log_metrics(
                phase="val",
                loss=loss,
                outputs=outputs,
                targets=targets,
                head_losses=head_losses,
                label_columns=label_columns,
                epoch=epoch,
                batch_idx=batch_idx
            )
        
        # TODO: Save results to file, print

def compute_and_log_metrics(phase, loss, outputs, targets, head_losses, label_columns, epoch, batch_idx):
    log = {
        "epoch": epoch,
        "batch": batch_idx,
        "loss": loss.item()
    }
    for output, target, head_loss, label_column in zip(outputs, targets, head_losses, label_columns):
        mask = target != -1   # target == -1 => sample has no output for this head
        output, target = output[mask], target[mask]  # filter out -1
        head_num_samples = output.shape[0]

        if head_num_samples:
            head_acc = utils.accuracy(output, target, topk=(1,))
            log[f"accuracy_{label_column}"] = head_acc.item()
            log[f"num_samples_{label_column}"] = head_num_samples

        log[f"loss_{label_column}"] = head_loss.item()

    wandb.log({
        f"{phase}/{key}": value
        for key, value in log
    })
    pprint(log)
    print()



def main():
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    cfg = load_config()

    utils.init_distributed_mode(cfg.distributed)

    # wandb
    wandb.login(host=cfg.wandb.url)
    wandb.init(project=cfg.wandb.project, config=cfg.to_dict(), notes=cfg.wandb.notes)

    device = torch.device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_dir = os.path.join(cfg.data_dir, cfg.train_subdir)
    valid_dir = os.path.join(cfg.data_dir, cfg.valid_subdir)

    print("Loading data")
    label_mappings = []
    for label_mapping_json in cfg.dataset.label_mapping_jsons:
        with open(label_mapping_json) as f:
            label_mapping = json.load(f)
            label_mappings.append(dict(zip(label_mapping, range(len(label_mapping)))))

    float_zero_to_one = torchvision.transforms.Lambda(lambda video: video.to(torch.float32) / 255)

    normalize = torchvision.transforms.Normalize(
        mean=[0.43216, 0.394666, 0.37645], 
        std=[0.22803, 0.22145, 0.216989])

    resize = torchvision.transforms.Resize((128, 171))

    train_transform = torchvision.transforms.Compose([
        float_zero_to_one,
        torchvision.transforms.Lambda(lambda video: video.permute(0, 3, 1, 2)),
        resize,
        torchvision.transforms.RandomHorizontalFlip(),
        normalize,
        torchvision.transforms.Lambda(lambda video: video.permute(1, 0, 2, 3)),
        torchvision.transforms.RandomCrop((112, 112))
    ])

    train_dataset = UntrimmedVideoDataset(
        csv_filename=cfg.dataset.train_csv_filename,
        root_dir=train_dir,
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        clips_per_segment=cfg.video.clips_per_segment,
        temporal_jittering=True,
        transforms=train_transform,
        label_columns=cfg.dataset.label_columns,
        label_mappings=label_mappings,
        global_video_features=cfg.tsp.global_video_features,
        debug=cfg.debug
    )

    valid_transform = torchvision.transforms.Compose([
        float_zero_to_one,
        torchvision.transforms.Lambda(lambda video: video.permute(0, 3, 1, 2)),
        resize,
        normalize,
        torchvision.transforms.Lambda(lambda video: video.permute(1, 0, 2, 3)),
        torchvision.transforms.CenterCrop((112, 112))
    ])

    valid_dataset = UntrimmedVideoDataset(
        csv_filename=cfg.dataset.valid_csv_filename,
        root_dir=valid_dir,
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        clips_per_segment=cfg.video.clips_per_segment,
        temporal_jittering=False,
        transforms=valid_transform,
        label_columns=cfg.dataset.label_columns,
        label_mappings=label_mappings,
        global_video_features=cfg.tsp.global_video_features,
        debug=cfg.debug
    )

    print("Creating dataloaders")
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True) if cfg.distributed.on else None
    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, shuffle=False) if cfg.distributed.on else None

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

    # Create backbone
    if cfg.tsp.backbone == 'vivit':
        model_official = timm.create_model(cfg.pretrained_models.vit, pretrained=True)
        model_official.eval()

      # Use return_preclassifier=True for VideoVisionTransformer
        feature_backbone = VideoVisionTransformer(model_official=model_official, **cfg.vivit)
        d_feat = feature_backbone.d_model
    else:
        raise NotImplementedError

    tsp_model = TSPModel(
        backbone=feature_backbone,
        d_feat=d_feat,
        num_tsp_classes=[len(l) for l in label_mappings],
        num_tsp_heads=len(cfg.dataset.label_columns),
        concat_gvf=cfg.tsp.global_video_features is not None,
    )

    tsp_model.to(device)
    if cfg.distributed.on and cfg.distributed.sync_bn:
        tsp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tsp_model)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # label == -1 => missing label

    if len(cfg.dataset.label_columns) == 1:
        fc_params = tsp_model.action_fc.parameters()
    elif len(cfg.dataset.label_columns) == 2:
        fc_params = chain(tsp_model.action_fc.parameters(), tsp_model.region_fc.parameters())
    else:
        raise NotImplementedError

    params = [
        {
            "params": tsp_model.feature_extractor.parameters(),
            "lr": cfg.tsp.backbone_lr * (cfg.distributed.world_size if cfg.distributed.on else 1),
            "name": "backbone"
        },
        {
            "params": fc_params,
            "lr": cfg.tsp.fc_lr * (cfg.distributed.world_size if cfg.distributed.on else 1),
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
    lr_milestones = [len(train_dataloader) * m for m in cfg.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=cfg.lr_gamma,
        warmup_iters=warmup_iters,
        warmup_factor=cfg.lr_warmup_factor
    )

    model_without_ddp = tsp_model
    if cfg.distributed.on:
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
        if cfg.distributed.on:
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

            # save by epoch
            utils.torch_save_on_master(
                checkpoint,
                os.path.join(cfg.output_dir, f"epoch_{epoch}.pth")
            )

            # latest
            utils.torch_save_on_master(
                checkpoint,
                os.path.join(cfg.output_dir, "checkpoint.pth")
            )

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

