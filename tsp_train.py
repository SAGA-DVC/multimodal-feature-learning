'''
Code adapted from https://github.com/HumamAlwassel/TSP
Alwassel, H., Giancola, S., & Ghanem, B. (2021). TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
'''


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
import numpy as np
from models.ast import AudioSpectrogramTransformer

from tsp.vivit_wrapper import VivitWrapper
from tsp.tsp_model import TSPModel, concat_combiner
from tsp.untrimmed_video_dataset import UntrimmedVideoDataset
from tsp.lr_scheduler import WarmupMultiStepLR
from tsp.config import load_config
from tsp import utils


def epoch_loop(model: TSPModel, criterion, optimizer, lr_scheduler, dataloader, device, epoch, print_freq, label_columns, loss_alphas, wandb_log):
    model.train()

    metric_logger = utils.MetricLogger(delimiter=' ')
    for g in optimizer.param_groups:
        metric_logger.add_meter(
            f'{g["name"]}-lr', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
        metric_logger.add_meter(
            'clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    header = f'Train Epoch {epoch}:'

    for (batch_idx, batch) in enumerate(metric_logger.log_every(dataloader, print_freq, header, device=device)):
        start_time = time.time()
        clip = {
            'video': batch['video'].to(device),
            'audio': batch['audio'].to(device)
        }

        # Global video feature (video + audio features)
        gvf = batch['gvf'].to(device) if 'gvf' in batch else None

        # targets has the class index directly, not one-hot-encoded
        # See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        targets = [batch[x].to(device)
                   for x in label_columns]  # [(B, 1), (B, 1)]

        # Forward pass through TSPModel
        outputs = model(clip, gvf=gvf)  # [(B, 2), (B, c)]

        # compute losses for each label column
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
            metric_logger=metric_logger,
            phase="train",
            loss=loss,
            outputs=outputs,
            targets=targets,
            head_losses=head_losses,
            label_columns=label_columns,
            epoch=epoch,
            batch_idx=batch_idx,
            wandb_log=wandb_log
        )

        for g in optimizer.param_groups:
            metric_logger.meters[f'{g["name"]}-lr'].update(g['lr'])
        metric_logger.meters['clips/s'].update(
            clip['video'].shape[0] / (time.time() - start_time))

        lr_scheduler.step()


def evaluate(model: TSPModel, criterion, dataloader, device, epoch, print_freq, label_columns, loss_alphas, output_dir, wandb_log):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter=' ')
    header = f'Valid Epoch {epoch}:'

    with torch.no_grad():
        for (batch_idx, batch) in enumerate(metric_logger.log_every(dataloader, print_freq, header, device=device)):
            clip = {
                'video': batch['video'].to(device, non_blocking=True),
                'audio': batch['audio'].to(device, non_blocking=True)
            }
            
            # Global video feature (video + audio features) 
            gvf = batch['gvf'].to(
                device, non_blocking=True) if 'gvf' in batch else None

            # Targets
            targets = [batch[x].to(device, non_blocking=True)
                       for x in label_columns]

            # Forward pass through model
            outputs = model(clip, gvf=gvf)

            # compute losses for each label column
            head_losses, loss = [], 0
            for output, target, alpha in zip(outputs, targets, loss_alphas):
                head_loss = criterion(output, target)
                head_losses.append(head_loss)
                loss += alpha * head_loss

            compute_and_log_metrics(
                metric_logger=metric_logger,
                phase="val",
                loss=loss,
                outputs=outputs,
                targets=targets,
                head_losses=head_losses,
                label_columns=label_columns,
                epoch=epoch,
                batch_idx=batch_idx,
                wandb_log=wandb_log
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    results = write_metrics_results_to_file(
        metric_logger, epoch, label_columns, output_dir)
    print(results)


def compute_and_log_metrics(metric_logger, phase, loss, outputs, targets, head_losses, label_columns, epoch, batch_idx, wandb_log=False):
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
            head_acc = utils.accuracy(output, target, topk=(1,))[0]
            log[f"accuracy-{label_column}"] = head_acc.item()
            log[f"num_samples-{label_column}"] = head_num_samples
            metric_logger.meters[f'acc-{label_column}'].update(
                head_acc.item(), n=head_num_samples)

        log[f"loss-{label_column}"] = head_loss.item()
        metric_logger.meters[f'loss-{label_column}'].update(head_loss.item())

    if wandb_log:
        wandb.log({
            f"{phase}/{key}": value
            for key, value in log.items()
        })

    pprint(log)
    print()

    metric_logger.update(loss=loss.item())


def write_metrics_results_to_file(metric_logger, epoch, label_columns, output_dir):
    results = f'** Valid Epoch {epoch}: '
    for label_column in label_columns:
        results += f' <{label_column}> Accuracy {metric_logger.meters[f"acc_{label_column}"].global_avg:.3f}'
        results += f' Loss {metric_logger.meters[f"loss_{label_column}"].global_avg:.3f};'

    results += f' Total Loss {metric_logger.meters["loss"].global_avg:.3f}'
    avg_acc = np.average(
        [metric_logger.meters[f'acc_{label_column}'].global_avg for label_column in label_columns])
    results += f' Avg Accuracy {avg_acc:.3f}'

    results = f'{results}\n'
    utils.write_to_file_on_master(file=os.path.join(output_dir, 'results.txt'),
                                  mode='a',
                                  content=results)

    return results


def main(cfg):
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    print(f"Training dataset CSV: {cfg.dataset.train_csv_filename}")
    print(f"Validation dataset CSV: {cfg.dataset.valid_csv_filename}")

    # Setup distributed processes (if enabled)
    utils.init_distributed_mode(cfg.distributed)

    # Setup wandb
    if cfg.wandb.on:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                   config=cfg.to_dict(), notes=cfg.wandb.notes)

    device = torch.device(cfg.device)

    os.makedirs(cfg.output_dir, exist_ok=True)
    train_dir = os.path.join(cfg.data_dir, cfg.train_subdir)
    valid_dir = os.path.join(cfg.data_dir, cfg.valid_subdir)

    print(f"Training dataset videos directory: {train_dir}")
    print(f"Validation dataset videos directory: {valid_dir}")

    label_mappings = []
    for label_mapping_json in cfg.dataset.label_mapping_jsons:
        with open(label_mapping_json) as f:
            label_mapping = json.load(f)
            label_mappings.append(
                dict(zip(label_mapping, range(len(label_mapping)))))


    print("Initializing datasets and dataloaders")

    # Video transforms
    float_zero_to_one = torchvision.transforms.Lambda(
        lambda video: video.to(torch.float32) / 255)

    normalize = torchvision.transforms.Normalize(
        mean=[0.43216, 0.394666, 0.37645],
        std=[0.22803, 0.22145, 0.216989])

    resize = torchvision.transforms.Resize(256)  # As used in ViViT

    train_video_transform = torchvision.transforms.Compose([
        float_zero_to_one,
        torchvision.transforms.Lambda(lambda video: video.permute(0, 3, 1, 2)),
        resize,
        torchvision.transforms.RandomHorizontalFlip(),
        normalize,
        torchvision.transforms.Lambda(lambda video: video.permute(1, 0, 2, 3)),
        torchvision.transforms.RandomCrop(
            (cfg.vivit.img_size, cfg.vivit.img_size))
    ])

    # Training dataset
    train_dataset = UntrimmedVideoDataset(
        csv_filename=cfg.dataset.train_csv_filename,
        root_dir=train_dir,
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        clips_per_segment=cfg.video.clips_per_segment,
        temporal_jittering=True,
        num_mel_bins=cfg.audio.num_mel_bins,
        audio_target_length=cfg.audio.target_length,
        video_transform=train_video_transform,
        label_columns=cfg.dataset.label_columns,
        label_mappings=label_mappings,
        global_video_features=cfg.tsp.global_video_features,
        debug=cfg.debug
    )

    valid_video_transform = torchvision.transforms.Compose([
        float_zero_to_one,
        torchvision.transforms.Lambda(lambda video: video.permute(0, 3, 1, 2)),
        resize,
        normalize,
        torchvision.transforms.Lambda(lambda video: video.permute(1, 0, 2, 3)),
        torchvision.transforms.CenterCrop(
            (cfg.vivit.img_size, cfg.vivit.img_size))
    ])

    # Validation dataset
    valid_dataset = UntrimmedVideoDataset(
        csv_filename=cfg.dataset.valid_csv_filename,
        root_dir=valid_dir,
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        clips_per_segment=cfg.video.clips_per_segment,
        temporal_jittering=False,
        num_mel_bins=cfg.audio.num_mel_bins,
        audio_target_length=cfg.audio.target_length,
        video_transform=valid_video_transform,
        label_columns=cfg.dataset.label_columns,
        label_mappings=label_mappings,
        global_video_features=cfg.tsp.global_video_features,
        debug=cfg.debug
    )


    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, shuffle=True) if cfg.distributed.on else None
    valid_sampler = torch.utils.data.DistributedSampler(
        valid_dataset, shuffle=False) if cfg.distributed.on else None

    # Dataloaders
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


    print("Creating TSP backbones")

    # Create backbones
    feature_backbones = []
    d_feats = []
    input_modalities = []
    if 'vivit' in cfg.tsp.backbones:
        print("Creating ViViT backbone")
        model_official = timm.create_model(
            cfg.pretrained_models.vit, pretrained=True)
        model_official.eval()

        # Use return_preclassifier=True for VideoVisionTransformer
        backbone = VivitWrapper(model_official=model_official, **cfg.vivit)
        feature_backbones.append(backbone)
        d_feats.append(backbone.d_model)
        input_modalities.append('video')

    if 'ast' in cfg.tsp.backbones:
        print("Creating AST backbone")
        model_official = timm.create_model(
            cfg.pretrained_models.ast, pretrained=cfg.ast.imagenet_pretrained)
        model_official.eval()

        backbone = AudioSpectrogramTransformer(
            model_official=model_official, **cfg.ast)
        feature_backbones.append(backbone)
        d_feats.append(backbone.d_model)
        input_modalities.append('audio')


    # Model to be trained
    tsp_model = TSPModel(
        backbones=feature_backbones,
        input_modalities=input_modalities,
        d_feats=d_feats,
        d_tsp_feat=d_feats[0],
        combiner=concat_combiner,
        num_tsp_classes=[len(l) for l in label_mappings],
        num_tsp_heads=len(cfg.dataset.label_columns),
        concat_gvf=cfg.tsp.global_video_features is not None,
    )


    tsp_model.to(device)
    if cfg.distributed.on and cfg.distributed.sync_bn:
        tsp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tsp_model)

    # Criterion for training (both heads)
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-1)  # label == -1 => missing label


    if len(cfg.dataset.label_columns) == 1:
        fc_params = tsp_model.action_fc.parameters()
    elif len(cfg.dataset.label_columns) == 2:
        fc_params = chain(tsp_model.action_fc.parameters(),
                          tsp_model.region_fc.parameters())
    else:
        raise NotImplementedError


    # TSPModel Parameters
    params = [
        *[{
            "params": backbone.parameters(),
            "lr": cfg.tsp.backbone_lr * (cfg.distributed.world_size if cfg.distributed.on else 1),
            "name": f"backbone_{i}"
        } for i, backbone in enumerate(tsp_model.backbones)],
        {
            "params": fc_params,
            "lr": cfg.tsp.fc_lr * (cfg.distributed.world_size if cfg.distributed.on else 1),
            "name": "fc"
        }
    ]

    # Optimizer
    optimizer = torch.optim.SGD(
        params,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )

    # Scheduler per iteration, not per epoch for warmup that lasts between epochs
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
        tsp_model = torch.nn.parallel.DistributedDataParallel(
            tsp_model, device_ids=[cfg.distributed.rank])
        model_without_ddp = tsp_model.module


    if cfg.resume:
        print(f'Resuming from checkpoint {cfg.resume}')
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint['epoch'] + 1


    # Only evaluate model on validation dataset
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
            output_dir=cfg.output_dir,
            wandb_log=cfg.wandb.on
        )
        return


    print("Starting training")
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed.on:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

        # One epoch
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
            loss_alphas=cfg.tsp.loss_alphas,
            wandb_log=cfg.wandb.on
        )


        # Checkpointing
        if cfg.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'cfg': cfg
            }

            # save by epoch, "epoch_x.pth"
            utils.torch_save_on_master(
                checkpoint,
                os.path.join(cfg.output_dir, f"epoch_{epoch}.pth")
            )

            # latest checkpoint is called "checkpoint.pth"
            utils.torch_save_on_master(
                checkpoint,
                os.path.join(cfg.output_dir, "checkpoint.pth")
            )


        if cfg.train_only_one_epoch:
            break
        else:
            # Validation
            evaluate(
                model=tsp_model,
                criterion=criterion,
                dataloader=valid_dataloader,
                device=device,
                epoch=epoch,
                print_freq=cfg.print_freq,
                label_columns=cfg.dataset.label_columns,
                loss_alphas=cfg.tsp.loss_alphas,
                output_dir=cfg.output_dir,
                wandb_log=cfg.wandb.on
            )


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == '__main__':
    cfg = load_config()
    main(cfg)
