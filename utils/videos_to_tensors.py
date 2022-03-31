import argparse
import os

import torch
import torchvision
from torchvision.io import read_video
import h5py

def main(args):
    if not any([args.video_h5, args.audio_h5]):
        print("Specify at least one of [--video-h5, --audio-h5]")
        return

    video_files = os.listdir(args.raw_video_dir)

    video_h5 = h5py.File(args.video_h5, "w")
    audio_h5 = h5py.File(args.audio_h5, "w")

    video_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(
            lambda video: video.to(torch.float32) / 255),
        torchvision.transforms.Lambda(lambda video: video.permute(0, 3, 1, 2)),
        torchvision.transforms.Resize(256),  # As used in ViViT
        torchvision.transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]),
        torchvision.transforms.Lambda(lambda video: video.permute(1, 0, 2, 3))])

    for (i, video_file) in enumerate(video_files):
        video_id = video_file.split(".")[0][2:]
        vframes, aframes, info = read_video(video_file)

        vframes = video_transforms(vframes)
        
        # TODO Verify whether this should be done here or for each clip separately
        # aframes = aframes_to_fbank(aframes, info['audio_fps'], 128, 1024)
        
        video_h5.create_dataset(video_id, data=vframes.numpy())
        audio_h5.create_dataset(video_id, data=aframes.numpy())

        print(f"Videos processed: {i+1} ")

    video_h5.close()
    audio_h5.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for reading raw videos"
                                     " and extracting the audio and video frame tensors")
    parser.add_argument("--raw-video-dir", type=str, required=True,
                        help="Path to directory which has the raw videos")

    parser.add_argument("--video-h5", type=str, required=False,
                        help="Path to store the h5 file for video frame tensors")

    parser.add_argument("--audio-h5", type=str, required=False,
                        help="Path to store the h5 file for audio frame tensors")
    main(parser.parse_args())
