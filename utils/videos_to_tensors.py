import os
import json
import sys
import time
from datetime import datetime
import re

import torch
# from torch.utils.data import Dataset, DataLoader
# import torch.distributed as dist
from torch.multiprocessing import Process, Pool, set_start_method, current_process
import torchvision
from torchvision.io import read_video
import h5py
from ml_collections import ConfigDict
import numpy as np

# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()

# def is_main_process():
#     return get_rank() == 0

# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_world_size()

# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True


# def print_master_only(is_master):
#     '''
#     This function disables printing when not in master process
#     '''
#     import builtins as __builtin__
#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)
    
#     __builtin__.print = print

# class VideoToTensors:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.init_distributed_mode(self.cfg.distributed)
#         self.video_h5 = h5py.File(f"{self.cfg.video_h5}_rank_{get_rank()}", "w")
#         self.audio_h5 = h5py.File(f"{self.cfg.audio_h5}_rank_{get_rank()}", "w")
#         self.done_video_list = []
#         if cfg.done_videos:
#             with open(cfg.done_videos, "r") as f:
#                 self.done_videos_list = json.load(f)


#     def init_distributed_mode(self, distributed_cfg):
#         if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#             distributed_cfg.rank = int(os.environ['RANK'])
#             distributed_cfg.world_size = int(os.environ['WORLD_SIZE'])
#             distributed_cfg.gpu = int(os.environ['LOCAL_RANK'])
#             if distributed_cfg.world_size == 1:
#                 print("Not using distributed mode")
#                 distributed_cfg = None
#                 return
#         else:
#             print("Not using distributed mode") 
#             distributed_cfg.world_size = 1
#             return

#         torch.cuda.set_device(distributed_cfg.gpu)
#         distributed_cfg.backend = 'nccl'

#         print(f'| distributed init (rank {distributed_cfg.rank}): {distributed_cfg.dist_url}', flush=True)

#         torch.distributed.init_process_group(
#             backend=distributed_cfg.backend,
#             init_method=distributed_cfg.dist_url,
#             world_size=distributed_cfg.world_size,
#             rank=distributed_cfg.rank
#         )

#     def main(self):
#         if not any([self.cfg.video_h5, self.cfg.audio_h5]):
#             print("Specify at least one of [--video-h5, --audio-h5]")
#             return



#         dataset = UntrimmedVideoDataset(self.cfg.raw_video_dir, self.done_video_list or [], debug=self.cfg.debug)

#         sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
#         dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, sampler=sampler, num_workers=self.cfg.num_workers, pin_memory=True)

#         video_transform = torchvision.transforms.Compose([
#             torchvision.transforms.Lambda(
#                 lambda video: video.to(torch.float32) / 255),
#             torchvision.transforms.Lambda(lambda video: video.permute(0, 1, 4, 2, 3)),  # (B, T, H, W, C) --> (B, T, C, H, W)
#             torchvision.transforms.Resize(256),  # As used in ViViT
#             torchvision.transforms.Normalize(
#                 mean=[0.43216, 0.394666, 0.37645],
#                 std=[0.22803, 0.22145, 0.216989]),
#             torchvision.transforms.Lambda(lambda video: video.permute(0, 2, 1, 3, 4))]) # (B, T, C, H, W) --> (B, C, T, H, W)

#         if cfg.distributed.on:
#             video_transform = torch.nn.parallel.DistributedDataParallel(video_transform, device_ids=[cfg.distributed.rank+2])
#         else:
#             raise NotImplementedError


#         for (batch_idx, batch) in enumerate(dataloader):

#             transformed_vframes = video_transforms(batch["vframes"])
            
#             for (video_id, vframes, aframes) in zip(batch["video_id"], transformed_vframes, batch["aframes"]):
#                 self.video_h5.create_dataset(video_id, data=vframes.numpy())
#                 self.audio_h5.create_dataset(video_id, data=aframes.numpy())

#             self.done_video_list.append(list(batch["video_id"]))
            
#             print(f"Batches processed: {batch_idx+1} ")
    
#     def cleanup(self):
#         self.video_h5.close()
#         self.audio_h5.close()

#         with open(f"{self.done_videos}_{datetime.now()}_rank_{get_rank()}", "w") as f:
#             json.dump(self.done_video_list, f)

#     def execute(self):
#         try:
#             self.main()
#         except KeyboardInterrupt:
#             self.cleanup()



# class UntrimmedVideoDataset(Dataset):
#     def __init__(self, videos_list):
        
#         self.video_list = videos_list

#     def __len__(self):
#         return len(self.video_list)

#     def __getitem__(self, idx):

#         filename = self.video_list[idx]
        
#         vframes, aframes, info = read_video(filename=filename)

#         return {
#             "video_id": filename.split('.')[0][2:],
#             "vframes": vframes,
#             "aframes": aframes,
#             "info": info
#         }


def worker(videos, rank):
    video_h5 = h5py.File(f"valh5/video_rank_{rank}.h5", "a")
    audio_h5 = h5py.File(f"valh5/audio_rank_{rank}.h5", "a")

    done_video_list = []
    failed_video_list = []
    video_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(
                lambda video: video.to(torch.float32) / 255),   
            torchvision.transforms.Lambda(lambda video: video.permute(0, 3, 1, 2)),  # (T, H, W, C) --> (T, C, H, W)
            torchvision.transforms.Resize(256),  # As used in ViViT
            torchvision.transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989]),
            torchvision.transforms.Lambda(lambda video: video.permute(1, 0, 2, 3))]) # (T, C, H, W) --> (C, T, H, W)

    try:
        for (i, video_file) in enumerate(videos):
            video_name = re.findall("v_.*\.mp4", video_file)[0]
            video_id = video_name.split(".")[0][2:]

            try:
                vframes, aframes, _ = read_video(video_file)

                vframes = video_transform(vframes.to(f"cuda:{rank+2}"))   # On GPU 2, 3, ...
                # vframes = video_transform(vframes)      # On CPU
            
                video_h5.create_dataset(video_id, data=vframes.cpu().numpy())
                audio_h5.create_dataset(video_id, data=aframes.cpu().numpy())
            except RuntimeError:
                failed_video_list.append(video_id)
                continue

            done_video_list.append(video_id)

            if i % 25 == 0:
                # Checkpoint
                print(f"Process {rank}: Video {i+1}/{len(videos)}")
                write_results(rank, done_video_list, failed_video_list)
                video_h5.close()
                audio_h5.close()
                video_h5 = h5py.File(f"valh5/video_rank_{rank}.h5", "a")
                audio_h5 = h5py.File(f"valh5/audio_rank_{rank}.h5", "a")

    except KeyboardInterrupt:
        raise
        
    finally:
        print("finally")
        write_results(rank, done_video_list, failed_video_list)
        video_h5.close()
        audio_h5.close()

    return done_video_list, failed_video_list


def write_results(rank, done_video_list, failed_video_list):
    with open(f"val-video-to-tensor-output/done_videos_{datetime.now()}_rank_{rank}.json", "w") as f:
        json.dump(done_video_list, f)
    with open(f"val-video-to-tensor-output/failed_videos_{datetime.now()}_rank_{rank}.json", "w") as f:
        json.dump(failed_video_list, f)    


# def batch_worker(videos, rank):
#     try:
#         video_h5 = h5py.File(f"video_rank_{rank}.h5", "w")
#         audio_h5 = h5py.File(f"audio_rank_{rank}.h5", "w")

#         done_video_list = []

#         dataset = UntrimmedVideoDataset(videos)

#         dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#         video_transform = torchvision.transforms.Compose([
#                 torchvision.transforms.Lambda(
#                     lambda video: video.to(torch.float32) / 255),   
#                 torchvision.transforms.Lambda(lambda video: video.permute(0, 1, 4, 2, 3)),  # (B, T, H, W, C) --> (B, T, C, H, W)
#                 torchvision.transforms.Resize(256),  # As used in ViViT
#                 torchvision.transforms.Normalize(
#                     mean=[0.43216, 0.394666, 0.37645],
#                     std=[0.22803, 0.22145, 0.216989]),
#                 torchvision.transforms.Lambda(lambda video: video.permute(0, 2, 1, 3, 4))]) # (B, T, C, H, W) --> (B, C, T, H, W)
            
#         for(batch_idx, batch) in enumerate(dataloader):
#             print(f"Loading batch {batch_idx}")
#             transformed_vframes = video_transform(batch["vframes"].to(f"cuda:{rank+2}"))
                
#             for (video_id, vframes, aframes) in zip(batch["video_id"], transformed_vframes, batch["aframes"]):
#                 video_h5.create_dataset(video_id, data=vframes.numpy())
#                 audio_h5.create_dataset(video_id, data=aframes.numpy())

#             done_video_list.append(list(batch["video_id"]))
            
#             print(f"Batches processed: {batch_idx+1} ")
#     except Exception as e:
#         print("Exception occured!")
#         print(e)
#     finally:
#         video_h5.close()
#         audio_h5.close()

#         with open(f"done_videos_{datetime.now()}_rank_{rank}.json", "w") as f:
#             json.dump(done_video_list, f)
#         return done_video_list


if __name__ == '__main__':

    print("Starting...")
    cfg = ConfigDict()

    cfg.raw_video_dir = '/home/arnavshah/activity-net/30fps_splits/val'
    cfg.done_videos = None
    # cfg.batch_size = 16
    # cfg.debug = True
    # if cfg.debug:
    #     cfg.batch_size=4

    # cfg.audio_h5 = "audio_features.h5"
    # cfg.video_h5 = "video_features.h5"

    cfg.num_workers = 2

    if cfg.done_videos:
        with open("cfg.done_videos", "r") as f:
            done_video_list = json.load(f)
    else:
        done_video_list = []

    videos = list(map(lambda v: f"{cfg.raw_video_dir}/{v}", os.listdir(cfg.raw_video_dir)))
    videos = [v for v in videos if v not in done_video_list]
    
    sub_size = len(videos) // cfg.num_workers

    print("Creating partitions...")
    partitions = []
    for i in range(cfg.num_workers):
        start = i * sub_size
        end = start + sub_size
        if i == cfg.num_workers - 1:
            end = len(videos)
        partitions.append(videos[start:end])
        
    print(len(partitions))
    
    set_start_method("spawn")

    processes = []

    for rank in range(cfg.num_workers):
        p = Process(target=worker, args=(partitions[rank], rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
