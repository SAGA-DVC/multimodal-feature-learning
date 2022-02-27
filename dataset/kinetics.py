from typing import Tuple
import torch
import pytorchvideo
from pytorchvideo import transforms
from pytorchvideo.data import Kinetics, make_clip_sampler
import torchvision


def get_kinetics(kinetics_root: str, num_temporal_samples: int, frame_size: Tuple[int, int], batch_size):

    transform = torchvision.transforms.Compose([
        transforms.RemoveKey("video_name"),
        transforms.RemoveKey("aug_index"),
        transforms.RemoveKey("audio"),
        transforms.ApplyTransformToKey(
            key="video",
            transform=torchvision.transforms.Compose([
                transforms.UniformTemporalSubsample(num_temporal_samples),
                transforms.Div255(),
                torchvision.transforms.CenterCrop(frame_size),
            ])
        )
    ])

    dataset = Kinetics(
        data_path=kinetics_root,
        clip_sampler=make_clip_sampler("random", 2),
        transform=transform
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataset, loader


if __name__ == '__main__':

    # just to check

    dataset, loader = get_kinetics(
        kinetics_root="../data/sample",
        num_temporal_samples=10,
        frame_size=(224, 224),
        batch_size=1
    )

    for i, batch in enumerate(iter(loader)):
        print(i, batch['video'].shape)
