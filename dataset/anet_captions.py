import json
from typing import Tuple
import torch
import pytorchvideo
from pytorchvideo import transforms
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
import torchvision


def get_anet_captions(anet_root: str, num_temporal_samples: int, frame_size: Tuple[int, int], batch_size):

    transform = torchvision.transforms.Compose([
        transforms.RemoveKey("video_name"),
        transforms.RemoveKey("aug_index"),
        transforms.RemoveKey("audio"),
        transforms.RemoveKey("annotation"),
        transforms.ApplyTransformToKey(
            key="video",
            transform=torchvision.transforms.Compose([
                transforms.UniformTemporalSubsample(num_temporal_samples),
                transforms.Div255(),
                torchvision.transforms.CenterCrop(frame_size),
            ])
        )
    ])

    video_path = f"{anet_root}/videos"

    with open(f"{anet_root}/available.json", "r") as f:
        available = json.load(f)

    labeled_video_paths = [
        (f"{video_path}/{k}.mp4", {"annotation": v}) for k, v in available.items()
    ]

    dataset = LabeledVideoDataset(
        labeled_video_paths=labeled_video_paths,
        clip_sampler=make_clip_sampler("random", 2),
        transform=transform
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataset, loader


if __name__ == '__main__':

    # just to check

    dataset, loader = get_anet_captions(
        anet_root="../data/anet",
        num_temporal_samples=10,
        frame_size=(224, 224),
        batch_size=3
    )

    for i, batch in enumerate(iter(loader)):
        print(i, batch['video'].shape)
