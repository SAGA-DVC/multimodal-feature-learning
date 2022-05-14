'''
Code adapted from https://github.com/HumamAlwassel/TSP
Alwassel, H., Giancola, S., & Ghanem, B. (2021). TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
'''


import datetime
from itertools import chain
import os
import time
import json

import torch
import torch.nn as nn
import torchvision
import wandb
import timm
import numpy as np

from tsp.vivit_wrapper import VivitWrapper
from models.ast import AudioSpectrogramTransformer
from tsp.video_cnn_backbones import i3d, r2plus1d_18, r2plus1d_34, r3d_18
from tsp.audio_cnn_backbones import vggish, PermuteAudioChannel
from tsp.tsp_model import TSPModel, add_combiner, concat_combiner
from tsp.untrimmed_video_dataset import UntrimmedVideoDataset
from tsp.config.tsp_train_config import load_config
from tsp import utils
from tsp.engine import epoch_loop, evaluate

def main(cfg):
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(0)
    print(f"Training dataset CSV: {cfg.dataset.train_csv_filename}")
    print(f"Validation dataset CSV: {cfg.dataset.valid_csv_filename}")

    assert len(cfg.tsp.backbones) <= 2, "Only two backbones supported yet, one for video and one for audio"

    print(f"Input modalities: {cfg.tsp.modalities}")
    print(f"Backbones: {cfg.tsp.backbones}")

    # Setup distributed processes (if enabled)
    utils.init_distributed_mode(cfg.distributed)

    # Setup wandb
    if cfg.wandb.on and utils.is_main_process():
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                   config=cfg.to_dict(), notes=cfg.wandb.notes)

    device = torch.device(cfg.device)
    
    print(f"Output directory: {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/grads", exist_ok=True)
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

    with open(cfg.dataset.unavailable_videos, "r") as f:
        unavailable_videos = json.load(f)

    # Training dataset
    print("Training Dataset:")
    train_dataset = UntrimmedVideoDataset(
        csv_filename=cfg.dataset.train_csv_filename,
        root_dir=train_dir,
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        clips_per_segment=cfg.video.clips_per_segment,
        temporal_jittering=True,
        modalities=cfg.tsp.modalities,
        num_mel_bins=cfg.audio.num_mel_bins,
        audio_target_length=cfg.audio.target_length,
        video_transform=train_video_transform,
        label_columns=cfg.dataset.label_columns,
        label_mappings=label_mappings,
        global_video_features=cfg.tsp.train_global_video_features,
        debug=cfg.debug,
        unavailable_videos=unavailable_videos
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

    print("Length of training dataset: ", len(train_dataset))

    # Validation dataset
    print("Validation Dataset:")
    valid_dataset = UntrimmedVideoDataset(
        csv_filename=cfg.dataset.valid_csv_filename,
        root_dir=valid_dir,
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        clips_per_segment=cfg.video.clips_per_segment,
        temporal_jittering=False,
        modalities=cfg.tsp.modalities,
        num_mel_bins=cfg.audio.num_mel_bins,
        audio_target_length=cfg.audio.target_length,
        video_transform=valid_video_transform,
        label_columns=cfg.dataset.label_columns,
        label_mappings=label_mappings,
        global_video_features=cfg.tsp.val_global_video_features,
        debug=cfg.debug,
        unavailable_videos=unavailable_videos
    )

    print("Length of validation dataset: ", len(valid_dataset))

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, shuffle=True) if cfg.distributed.on else torch.utils.data.RandomSampler(train_dataset)
    valid_sampler = torch.utils.data.DistributedSampler(
        valid_dataset, shuffle=True) if cfg.distributed.on else torch.utils.data.RandomSampler(valid_dataset)

    # Dataloaders

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.val_batch_size,
        sampler=valid_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )


    print("Creating TSP backbones:")

    # Create backbones
    feature_backbones = []
    d_feats = []

    # Video Backbone
    if 'vivit' in cfg.tsp.backbones:
        print("Creating ViViT backbone")

        # Use return_prelogits=True for VideoVisionTransformer
        backbone = VivitWrapper(**cfg.vivit)

        if cfg.pretrained_models.vivit:
            state_dict = torch.load(cfg.pretrained_models.vivit)
            backbone.load_weights_from_state_dict(state_dict)

        backbone.to(device)
        if cfg.vivit_freeze_first_n_encoder_blocks:
            for i in range(cfg.vivit_freeze_first_n_encoder_blocks):
                backbone.vivit.vivitEncoder.encoder[i].requires_grad_(False)
        feature_backbones.append(backbone)
        d_feats.append(backbone.d_model)
    
    elif 'r2plus1d_34' in cfg.tsp.backbones:
        print("Creating R(2+1)D-34 backbone")
        backbone = r2plus1d_34(pretrained=True)
        d_feats.append(backbone.fc.in_features)
        backbone.fc = nn.Sequential()
        backbone.to(device)
        feature_backbones.append(backbone)
        
    elif 'r2plus1d_18' in cfg.tsp.backbones:
        print("Creating R(2+1)D-18 backbone")
        backbone = r2plus1d_18(pretrained=True)
        d_feats.append(backbone.fc.in_features)
        backbone.fc = nn.Sequential()
        backbone.to(device)
        feature_backbones.append(backbone)
        
    elif 'r3d_18' in cfg.tsp.backbones:
        print("Creating R3D-18 backbone")
        backbone = r3d_18(pretrained=True)
        d_feats.append(backbone.fc.in_features)
        backbone.fc = nn.Sequential()
        backbone.to(device)
        feature_backbones.append(backbone)
    
    elif 'i3d' in cfg.tsp.backbones:
        print("Creating I3D backbone")
        backbone = i3d(pretrained=True)
        d_feats.append(backbone.blocks[-1].proj.in_features)
        backbone.blocks[-1].proj = torch.nn.Identity()
        backbone.to(device)
        feature_backbones.append(backbone)


    # Audio Backbone
    if 'ast' in cfg.tsp.backbones:
        print("Creating AST backbone")
        model_official = timm.create_model(
            cfg.pretrained_models.ast, pretrained=True)
        model_official.eval()

        backbone = AudioSpectrogramTransformer(
            model_official=model_official, **cfg.ast)
        
        if cfg.pretrained_models.ast_audioset:
                state_dict = torch.load(cfg.pretrained_models.ast_audioset)
                backbone.load_state_dict(state_dict)

        backbone.to(device)
        d_feats.append(backbone.d_model)
        feature_backbones.append(backbone)

    elif 'vggish' in cfg.tsp.backbones:
        # Requires:
        # cfg.audio.num_mel_bins = 64
        # cfg.audio.target_length = 96
        print("Creating VGGish backbone")
        backbone = nn.Sequential(
            PermuteAudioChannel(),
            vggish(pretrained=True, device=device)
        )
        backbone.to(device)
        d_feats.append(backbone[1].embeddings[-2].out_features)
        feature_backbones.append(backbone)

    total_params = 0
    for (modality, backbone_name, backbone) in zip(cfg.tsp.modalities, cfg.tsp.backbones, feature_backbones):
        n_parameters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f'Number of trainable params in {modality} backbone {backbone_name}: {n_parameters / 1e6} M')
        total_params += n_parameters

    # Model to be trained
    tsp_model = TSPModel(
        backbones=feature_backbones,
        input_modalities=cfg.tsp.modalities,
        d_feat=sum(d_feats),
        d_tsp_feat=512,
        combiner=concat_combiner,
        num_tsp_classes=[len(l) for l in label_mappings],
        num_tsp_heads=len(cfg.dataset.label_columns),
        concat_gvf=cfg.tsp.train_global_video_features is not None,
    )

    total_params += sum(p.numel() for p in tsp_model.parameters() if p.requires_grad)
    print(f'Total number of trainable params: {(total_params) / 1e6} M')

    tsp_model.to(device)
    if cfg.distributed.on and cfg.distributed.sync_bn:
        tsp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tsp_model)

    # Criterion for training (both heads)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # label == -1 => missing label


    if len(cfg.dataset.label_columns) == 1:
        fc_params = chain(tsp_model.fc.parameters(), 
                            tsp_model.action_fc.parameters())
    elif len(cfg.dataset.label_columns) == 2:
        fc_params = chain(tsp_model.fc.parameters(),
                            tsp_model.action_fc.parameters(),
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

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

    model_without_ddp = tsp_model
    if cfg.distributed.on:
        tsp_model = torch.nn.parallel.DistributedDataParallel(
            tsp_model, device_ids=[cfg.distributed.rank], find_unused_parameters=True)
        model_without_ddp = tsp_model.module

    if utils.is_main_process() and cfg.wandb.on:
        wandb.watch(model_without_ddp, log_freq=100, log='all')
        for (i, backbone) in enumerate(cfg.tsp.backbones):
            wandb.watch(model_without_ddp.backbones[i], log_freq=100, log='all')

    if cfg.resume:
        print(f'Resuming from checkpoint {cfg.resume}')
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        for (i, backbone) in enumerate(cfg.tsp.backbones):
            model_without_ddp.backbones[i].load_state_dict(checkpoint[backbone])
            print(f"Loaded {backbone} weights from checkpoint")

        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint['epoch'] + 1

    print("Start epoch: ", cfg.start_epoch)


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
            wandb_log=cfg.wandb.on,
            output_dir=cfg.output_dir,
            plot_grads=cfg.plot_grads
        )


        # Checkpointing
        if cfg.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'cfg': cfg
            }
            if lr_scheduler:
                checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

            for (key, backbone) in zip(cfg.tsp.backbones, model_without_ddp.backbones):
                checkpoint[key] = backbone.state_dict()

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


            if cfg.wandb.on and utils.is_main_process():
                # versioning on wandb
                artifact = wandb.Artifact("tsp", type="model", description=f"tsp model with backbones: {cfg.tsp.backbones}")
                artifact.add_file(os.path.join(cfg.output_dir, f"epoch_{epoch}.pth"))
                wandb.log_artifact(artifact)


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
