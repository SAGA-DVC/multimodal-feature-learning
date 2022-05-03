'''
Code adapted from https://github.com/HumamAlwassel/TSP
Alwassel, H., Giancola, S., & Ghanem, B. (2021). TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
'''

from email.charset import add_charset
import os
import sys
import json

import timm
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd

sys.path.insert(0, '..')

from tsp.config.extract_features_config import load_config
from tsp.eval_video_dataset import EvalVideoDataset
from tsp.tsp_model import TSPModel, add_combiner, concat_combiner
from tsp import utils
from tsp.vivit_wrapper import VivitWrapper
from models.ast import AudioSpectrogramTransformer
from tsp.video_cnn_backbones import i3d, r2plus1d_18, r2plus1d_34, r3d_18
from tsp.audio_cnn_backbones import vggish, PermuteAudioChannel


def evaluate(model, dataloader, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter=' ')
    header = 'Feature extraction:'

    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_logger.log_every(dataloader, 10, header, device=device)):
            clip = {}

            # Modalities
            if 'video' in batch['clip']:
                clip['video'] = batch['clip']['video'].to(device, non_blocking=True)
            if 'audio' in batch['clip']:
                clip['audio'] = batch['clip']['audio'].to(device, non_blocking=True)

            # Forward pass through the model
            features = model(clip)
            # Save features as pkl (if features of all clips of a video have been collected)
            dataloader.dataset.save_features(features, batch)

            print(f"Batch: {batch_idx}")


def main(cfg):
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    print(f'Using device: {cfg.device}')

    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.device)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Video Transforms
    video_transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda video: video.to(torch.float32) / 255),
        torchvision.transforms.Lambda(lambda video: video.permute(0, 3, 1, 2)),
        torchvision.transforms.Resize(256),
        torchvision.transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645], 
            std=[0.22803, 0.22145, 0.216989]),
        torchvision.transforms.Lambda(lambda video: video.permute(1, 0, 2, 3)),
        torchvision.transforms.CenterCrop((cfg.vivit.img_size, cfg.vivit.img_size))
    ])

    metadata_df = pd.read_csv(cfg.metadata_csv_filename)

    # Shards for parallel processing
    shards = np.linspace(0,len(metadata_df),cfg.num_shards+1).astype(int)

    # Start and end idxs for current process
    start_idx, end_idx = shards[cfg.shard_id], shards[cfg.shard_id+1]
    print(f'shard: {cfg.shard_id + 1} of {cfg.num_shards}, (ID: {cfg.shard_id}) '
        f'total number of videos: {len(metadata_df)}, shard size {end_idx-start_idx} videos')

    # Keep current process' shard only
    metadata_df = metadata_df.iloc[start_idx:end_idx].reset_index()

    # Mark those videos whose pkl features are present already
    metadata_df['is-computed-already'] = metadata_df['filename'].map(lambda f:
        os.path.exists(os.path.join(cfg.output_dir, os.path.basename(f).split('.')[0] + '.pkl')))

    # Drop those videos whose pkl features are present already
    metadata_df = metadata_df[metadata_df['is-computed-already']==False].reset_index(drop=True)
    print(f'Number of videos to process after excluding the ones already computed on disk: {len(metadata_df)}')

    with open(cfg.dataset.unavailable_videos, "r") as f:
        unavailable_videos = json.load(f)

    print(f"Using modalities: {cfg.tsp.modalities}")
    print(f"Using backbones: {cfg.tsp.backbones}")

    # Dataset
    dataset = EvalVideoDataset(
        metadata_df=metadata_df,
        root_dir=f'{cfg.data_dir}/{cfg.subdir}',
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        stride=cfg.video.stride,
        output_dir=cfg.output_dir,
        num_mel_bins=cfg.audio.num_mel_bins,
        audio_target_length=cfg.audio.target_length,
        modalities=cfg.tsp.modalities,
        video_transform=video_transform,
        unavailable_videos=unavailable_videos
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    # Create backbones
    feature_backbones = []
    d_feats = []

    # Video Backbone
    if 'vivit' in cfg.tsp.backbones:
        print("Creating ViViT backbone")

        # Use return_preclassifier=True for VideoVisionTransformer
        backbone = VivitWrapper(**cfg.vivit)
        if cfg.pretrained_models.vivit:
            state_dict = torch.load(cfg.pretrained_models.vivit)
            backbone.load_weights_from_state_dict(state_dict)

        backbone = backbone.to(device)
        feature_backbones.append(backbone)
        d_feats.append(backbone.d_model)
    
    elif 'r2plus1d_34' in cfg.tsp.backbones:
        print("Creating R(2+1)D-34 backbone")
        backbone = r2plus1d_34(pretrained=False)
        d_feats.append(backbone.fc.in_features)
        backbone.fc = nn.Sequential()

        if cfg.r2plus1d_34_weights:
            print(f"Using TSP pretrained weights for R(2+1)D-34 from {cfg.r2plus1d_34_weights}")
            pretrained_weights = torch.load(cfg.r2plus1d_34_weights)
            backbone.load_state_dict(pretrained_weights)

        backbone.to(device)
        feature_backbones.append(backbone)
        
    elif 'r2plus1d_18' in cfg.tsp.backbones:
        print("Creating R(2+1)D-18 backbone")
        backbone = r2plus1d_18(pretrained=False)
        d_feats.append(backbone.fc.in_features)
        backbone.fc = nn.Sequential()

        if cfg.r2plus1d_18_weights:
            print(f"Using TSP pretrained weights for R(2+1)D-18 from {cfg.r2plus1d_18_weights}")
            pretrained_weights = torch.load(cfg.r2plus1d_18_weights)
            backbone.load_state_dict(pretrained_weights)

        backbone.to(device)
        feature_backbones.append(backbone)
        
    elif 'r3d_18' in cfg.tsp.backbones:
        print("Creating R3D-18 backbone")
        backbone = r3d_18(pretrained=False)
        d_feats.append(backbone.fc.in_features)
        backbone.fc = nn.Sequential()

        if cfg.r3d_18_weights:
            print(f"Using TSP pretrained weights for R3D-18 from {cfg.r3d_18_weights}")
            pretrained_weights = torch.load(cfg.r3d_18_weights)
            backbone.load_state_dict(pretrained_weights)

        backbone.to(device)
        feature_backbones.append(backbone)
    
    elif 'i3d' in cfg.tsp.backbones:
        print("Creating I3D backbone")
        backbone = i3d(pretrained=False)
        d_feats.append(backbone.blocks[-1].proj.in_features)
        backbone.blocks[-1].proj = torch.nn.Identity()

        if cfg.i3d_weights:
            print(f"Using TSP pretrained weights for I3D from {cfg.i3d_weights}")
            pretrained_weights = torch.load(cfg.i3d_weights)
            backbone.load_state_dict(pretrained_weights)

        backbone.to(device)
        feature_backbones.append(backbone)
    
    
    # Audio Backbone
    if 'ast' in cfg.tsp.backbones:
        print("Creating AST backbone")
        model_official = timm.create_model(cfg.pretrained_models.ast, pretrained=cfg.ast.imagenet_pretrained)
        model_official.eval()

        backbone = AudioSpectrogramTransformer(model_official=model_official, **cfg.ast)

        if cfg.pretrained_models.ast_audioset:
            state_dict = torch.load(cfg.pretrained_models.ast_audioset)
            backbone.load_state_dict(state_dict)

        backbone = backbone.to(device)
        feature_backbones.append(backbone)
        d_feats.append(backbone.d_model)

    elif 'vggish' in cfg.tsp.backbones:
        # Requires:
        # cfg.audio.num_mel_bins = 64
        # cfg.audio.target_length = 96
        print("Creating VGGish backbone")
        backbone = nn.Sequential(
            PermuteAudioChannel(),
            vggish(pretrained=False, device=device)
        )

        if cfg.vggish_weights:
            print(f"Using TSP pretrained weights for VGGish from {cfg.vggish_weights}")
            pretrained_weights = torch.load(cfg.vggish_weights)
            backbone.load_state_dict(pretrained_weights)


        backbone.to(device)
        d_feats.append(backbone[1].embeddings[-2].out_features)
        feature_backbones.append(backbone)


    # model with a dummy classifier layer
    tsp_model = TSPModel(
        backbones=feature_backbones,
        input_modalities=cfg.tsp.modalities,
        d_feat=sum(d_feats),
        d_tsp_feat=512,
        combiner=concat_combiner,
        num_tsp_classes=[],
        num_tsp_heads=0,
        concat_gvf=False,
    )

    if cfg.local_checkpoint:
        checkpoint = torch.load(cfg.local_checkpoint, map_location='cpu')

        tsp_model.fc.load_state_dict(checkpoint['model'], strict=False)  # strict=False to ignore keys of TSP heads

        # Load backbone weights from checkpoint
        for (i, backbone) in enumerate(cfg.tsp.backbones):
            tsp_model.backbones[i].load_state_dict(checkpoint[backbone])
            print(f"Loaded {backbone} weights from checkpoint")

    tsp_model.to(device)

    print('Starting feature extraction')
    evaluate(tsp_model, dataloader, device)


if __name__ == '__main__':
    cfg = load_config()
    main(cfg)