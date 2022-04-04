'''
Code adapted from https://github.com/HumamAlwassel/TSP
Alwassel, H., Giancola, S., & Ghanem, B. (2021). TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
'''

import os

import timm
import torch
import torchvision
import numpy as np
import pandas as pd

from tsp.config import load_config
from tsp.eval_video_dataset import EvalVideoDataset
from tsp.tsp_model import TSPModel, concat_combiner
from tsp import utils
from tsp.vivit_wrapper import VivitWrapper
from models.ast import AudioSpectrogramTransformer


def evaluate(model, dataloader, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter=' ')
    header = 'Feature extraction:'

    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_logger.log_every(dataloader, 10, header, device=device)):
            clip = {
                "video": batch['clip']['video'].to(device, non_blocking=True),
                "audio": batch['clip']['audio'].to(device, non_blocking=True)
            }

            # Forward pass through the model
            _, features = model(clip, return_features=True)
            # Save features as pkl (if features of all clips of a video have been collected)
            dataloader.dataset.save_features(features, batch)

            print(f"Batch: {batch_idx}")


def main(cfg):
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)

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
    print(f'shard-id: {cfg.shard_id + 1} out of {cfg.num_shards}, '
        f'total number of videos: {len(metadata_df)}, shard size {end_idx-start_idx} videos')

    # Keep current process' shard only
    metadata_df = metadata_df.iloc[start_idx:end_idx].reset_index()

    # Mark those videos whose pkl features are present already
    metadata_df['is-computed-already'] = metadata_df['filename'].map(lambda f:
        os.path.exists(os.path.join(cfg.output_dir, os.path.basename(f).split('.')[0] + '.pkl')))

    # Drop those videos whose pkl features are present already
    metadata_df = metadata_df[metadata_df['is-computed-already']==False].reset_index(drop=True)
    print(f'Number of videos to process after excluding the ones already computed on disk: {len(metadata_df)}')

    # Dataset
    dataset = EvalVideoDataset(
        metadata_df=metadata_df,
        root_dir=f'{cfg.data_dir}/{cfg.feature_extraction.subdir}',
        clip_length=cfg.video.clip_len,
        frame_rate=cfg.video.frame_rate,
        stride=cfg.video.stride,
        output_dir=cfg.output_dir,
        num_mel_bins=cfg.audio.num_mel_bins,
        audio_target_length=cfg.audio.target_length,
        video_transform=video_transform
    )

    print('CREATING DATA LOADER')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    # Create backbones
    feature_backbones = []
    d_feats = []
    input_modalities = []
    if 'vivit' in cfg.tsp.backbones:
        print("Creating ViViT backbone")
        model_official = timm.create_model(cfg.pretrained_models.vit, pretrained=True)
        model_official.eval()

        # Use return_preclassifier=True for VideoVisionTransformer
        backbone = VivitWrapper(model_official=model_official, **cfg.vivit)
        feature_backbones.append(backbone)
        d_feats.append(backbone.d_model)
        input_modalities.append('video')
    
    if 'ast' in cfg.tsp.backbones:
        print("Creating AST backbone")
        model_official = timm.create_model(cfg.pretrained_models.ast, pretrained=cfg.ast.imagenet_pretrained)
        model_official.eval()

        backbone = AudioSpectrogramTransformer(model_official=model_official, **cfg.ast)
        feature_backbones.append(backbone)
        d_feats.append(backbone.d_model)
        input_modalities.append('audio')


    # model with a dummy classifier layer
    model = TSPModel(
        backbones=feature_backbones,
        input_modalities=input_modalities,
        d_feats=d_feats,
        d_tsp_feat=d_feats[0],
        num_tsp_classes=[1],
        num_tsp_heads=1, 
        concat_gvf=False,
        combiner=concat_combiner
    )

    # Resume from local checkpoint
    if cfg.local_checkpoint:
        print(f'Resuming from the local checkpoint: {cfg.local_checkpoint}')
        pretrained_state_dict = torch.load(cfg.local_checkpoint, map_location='cpu')['model']
        # remove the classifier layers from the pretrained model and load the backbone weights
        pretrained_state_dict = {k: v for k,v in pretrained_state_dict.items() if 'fc' not in k}
        state_dict = model.state_dict()
        pretrained_state_dict['fc.weight'] = state_dict['fc.weight']
        pretrained_state_dict['fc.bias'] = state_dict['fc.bias']
        model.load_state_dict(pretrained_state_dict)

    model.to(device)

    print('START FEATURE EXTRACTION')
    evaluate(model, dataloader, device)


if __name__ == '__main__':
    cfg = load_config()
    main(cfg)