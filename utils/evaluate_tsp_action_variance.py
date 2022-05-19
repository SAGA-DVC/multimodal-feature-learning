import argparse
import os
import json

import torch
import h5py
from tqdm import tqdm
import numpy as np
import scipy.stats as stats


def cosine_similarity_matrix(x1: torch.Tensor, x2:torch.Tensor=None, eps=1e-8) -> torch.Tensor:
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def main(args):
    if not os.path.exists(args.tsp_h5):
        print(f"Invalid path for TSP h5, {args.tsp_h5} does not exist")
        return

    if not os.path.exists(args.pre_h5):
        print(f"Invalid path for pre-TSP h5, {args.pre_h5} does not exist")
        return

    if not os.path.exists(args.annotations_file):
        print(f"Invalid path for annotations-file , {args.annotations_file} does not exist")
        return
    
    tsp_h5 = h5py.File(args.tsp_h5, "r")
    pre_h5 = h5py.File(args.pre_h5, "r")
    with open(args.annotations_file, "r") as f:
        annotations_file = json.load(f)['database']

    pre_action_var_list = []
    tsp_action_var_list = []
    pre_sum_action_var = 0.
    tsp_sum_action_var = 0.
    sum_var_diff = 0.
    num_segments = 0
    errors = 0

    for video in tqdm(tsp_h5.keys()):
        try:
            pre_features = pre_h5[video]
            tsp_features = tsp_h5[video]
        except KeyError:
            continue

        pre_sim_matrix = cosine_similarity_matrix(torch.tensor(np.array(pre_features)))
        tsp_sim_matrix = cosine_similarity_matrix(torch.tensor(np.array(tsp_features)))

        annotations = annotations_file[video[2:]]['annotations']

        # For each annotated segment (action segment)
        for annotation in annotations:
            start, end = annotation['segment']
            
            start = int((start * args.fps) / args.frames_per_clip)
            end = int((end * args.fps) / args.frames_per_clip)
            end = min(pre_sim_matrix.shape[0]-1, end+1) if end == start else end

            pre_action_segment = pre_sim_matrix[start:end, start:end]
            tsp_action_segment = tsp_sim_matrix[start:end, start:end]

            if(pre_action_segment.numel() == 0 or tsp_action_segment.numel() == 0):
                errors += 1
                print(f"Error in video {video} segment start={start} end={end}. Numel = 0")
                continue

            pre_action_var = pre_action_segment.var().item()
            tsp_action_var = tsp_action_segment.var().item()

            if(np.isnan(pre_action_var) or np.isnan(tsp_action_var)):
                print(f"Error in video {video} segment start={start} end={end}")
                errors += 1
                continue

            pre_action_var_list.append(pre_action_var)
            tsp_action_var_list.append(tsp_action_var)

            pre_sum_action_var += pre_action_var
            tsp_sum_action_var += tsp_action_var

            sum_var_diff += abs(pre_action_var - tsp_action_var)

            num_segments += 1
        
    pre_avg_action_var = pre_sum_action_var / num_segments
    tsp_avg_action_var = tsp_sum_action_var / num_segments
    avg_var_diff = sum_var_diff / num_segments

    print(f"Avg Pre-TSP action segments similarity variance: {pre_avg_action_var}")
    print(f"Avg TSP action segments similarity variance: {tsp_avg_action_var}")
    print(f"Avg difference between action segment similarity variance: {avg_var_diff}")
    print(f"Error in processing {errors} segments")

    print(f"T-score: {stats.ttest_rel(pre_action_var_list, tsp_action_var_list)}")

    with open("pre-action-var.json", "w") as f:
        json.dump(pre_action_var_list, f)
    
    with open("tsp-action-var.json", "w") as f:
        json.dump(tsp_action_var_list, f)

    pre_h5.close()
    tsp_h5.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to calculate a `voodoo`"
        " metric for evaluating the temporal sensitivity of video features")

    parser.add_argument("--tsp-h5", type=str, required=True, help="TSP features h5 file")
    parser.add_argument("--pre-h5", type=str, required=True, help="Pre-TSP features h5 file")
    parser.add_argument("--annotations-file", type=str, help="Annotations JSON file",
        default="/home/arnavshah/tsp/tsp/dataset/activity_net.v1-3.min.json")
    parser.add_argument("--fps", type=int, default=30, help="FPS used while extraction")
    parser.add_argument("--frames-per-clip", type=int, default=16, help="Frames per clip")

    args = parser.parse_args()

    main(args)
