import argparse
import os
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
import time
import json

import torch
import h5py
from tqdm import tqdm
import numpy as np


def cosine_similarity_matrix(x1: torch.Tensor, x2:torch.Tensor=None, eps=1e-8) -> torch.Tensor:
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def main(args):
    if not os.path.exists(args.h5):
        print(f"Invalid path for h5, {args.h5} does not exist")
        return
    
    if not os.path.exists(args.annotations_file):
        print(f"Invalid path for annotations-file , {args.annotations_file} does not exist")
        return
    
    h5 = h5py.File(args.h5, "r")
    with open(args.annotations_file, "r") as f:
        annotations_file = json.load(f)['database']

    metric = 0.
    errors = 0

    for (idx, video) in enumerate(tqdm(h5.keys())):
        features = h5[video]

        sim_matrix = cosine_similarity_matrix(torch.tensor(np.array(features)))

        total_numel = sim_matrix.numel()
        total_sum = sim_matrix.sum()

        action_numel = 0
        action_sum = 0.

        annotations = annotations_file[video[2:]]['annotations']

        # For each annotated segment (action segment)
        for annotation in annotations:
            start, end = annotation['segment']
            
            start = int((start * args.fps) / args.frames_per_clip)
            end = int((end * args.fps) / args.frames_per_clip)
            action_segment = sim_matrix[start:end, start:end]

            action_numel += action_segment.numel()
            action_sum += action_segment.sum()
        
        non_action_numel = total_numel - action_numel
        non_action_sum = total_sum - action_sum

        if non_action_numel == 0:
            # metric += 0
            continue

        if action_numel == 0:
            print(f"Video {video} has action_numel = 0, last start={start} end={end}, action_segment: {action_segment.shape}")
            errors += 1
            print("Continuing")
            continue


        action_avg = action_sum / action_numel
        non_action_avg = non_action_sum / non_action_numel


        # Absolute difference 
        metric += abs(action_avg - non_action_avg)

    print(f"Mean absolute difference = {metric / idx}")
    print(f"Errors in processing {errors} videos")

    h5.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to calculate a `voodoo`"
        " metric for evaluating the temporal sensitivity of video features")

    parser.add_argument("--h5", type=str, required=True, help="TSP features h5 file")
    # parser.add_argument("--set", type=str, choices=["train", "training", "val", 
    #     "validation", "test", "testing"])
    parser.add_argument("--annotations-file", type=str, help="Annotations JSON file",
        default="/home/arnavshah/tsp/tsp/dataset/activity_net.v1-3.min.json")
    parser.add_argument("--fps", type=int, default=30, help="FPS used while extraction")
    parser.add_argument("--frames-per-clip", type=int, default=16, help="Frames per clip")

    args = parser.parse_args()

    main(args)
