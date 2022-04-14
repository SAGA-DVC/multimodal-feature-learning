'''
Codr adapted from https://github.com/HumamAlwassel/TSP
Alwassel, H., Giancola, S., & Ghanem, B. (2021). TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
'''


import os
import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import torch
import h5py
import pickle as pkl

from torch.utils.data import Dataset
from torchvision.io import read_video
from models.ast_utils import aframes_to_fbank

class EvalVideoDataset(Dataset):
    '''
    EvalVideoDataset:
        This dataset takes in a list of videos and return all clips with the given length and stride
        Each item in the dataset is a dictionary with the keys:
            - "clip": a dictionary with video and audio Tensors (dtype=torch.float)
              of the clip frames after applying transforms
            - "filename": the video filename
            - "is-last-clip": a flag to mark the last clip in the video
    '''

    def __init__(self, metadata_df, root_dir, clip_length, frame_rate, stride, output_dir, 
        num_mel_bins, audio_target_length, video_transform=None, unavailable_videos=[]):
        '''
        Args:
            metadata_df (pandas.DataFrame): a DataFrame with the following video metadata columns:
                [filename, fps, video-frames].
            root_dir (string): Directory with all the video files.
            clip_length (int): The number of frames per clip.
            frame_rate (int): The effective frame rate (fps) to sample clips.
            stride (int): The number of frames (after resampling with frame_rate) between consecutive clips.
                For example, `stride`=1 will generate dense clips, while `stride`=`clip_length` will generate non-overlapping clips
            output_dir (string): Path to the directory where video features will be saved
            video_transform (callable): A function/transform that takes in a TxHxWxC video
                and returns a transformed version.
            num_mel_bins (int) TODO
            audio_target_length TODO
            unavailable_videos: (List[str]): A list of unavailable videos to filter out (with file extension)
        '''
        metadata_df = EvalVideoDataset._remove_unavailable_raw_videos_from_df(metadata_df, root_dir, unavailable_videos)
        metadata_df = EvalVideoDataset._make_filenames_absolute(metadata_df, root_dir)
        EvalVideoDataset._check_files_exist(metadata_df)

        self.clip_metadata_df = EvalVideoDataset._generate_clips_metadata(metadata_df, clip_length, frame_rate, stride)
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.stride = stride
        self.output_dir = output_dir
        self.video_transform = video_transform

        self.audio_target_length = audio_target_length
        self.num_mel_bins = num_mel_bins

        # Holds clip features for a given video until all clips are processed and the
        # full video features are ready to be saved to disk
        self.saved_features = {}

    def __len__(self):
        return len(self.clip_metadata_df)

    def __getitem__(self, idx):
        sample = {}
        row = self.clip_metadata_df.iloc[idx]
        filename, fps = row['filename'], row['fps']

        filename, fps, clip_t_start, is_last_clip = row['filename'], row['fps'], row['clip-t-start'], row['is-last-clip']

        # compute clip_t_start and clip_t_end
        clip_length_in_sec = self.clip_length / self.frame_rate
        clip_t_end = clip_t_start + clip_length_in_sec
        # get a tensor [clip_length, H, W, C] of the video frames between clip_t_start and clip_t_end seconds
        vframes, aframes, info = read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
        idxs = EvalVideoDataset._resample_video_idx(self.clip_length, fps, self.frame_rate)
        vframes = vframes[idxs][:self.clip_length] # [:self.clip_length] for removing extra frames if isinstance(idxs, slice)
        if vframes.shape[0] != self.clip_length:
            # TODO
            # Temp fix, not sure whether this is the right way
            vframes = torch.cat((vframes, torch.zeros(self.clip_length - vframes.shape[0], *vframes.shape[1:])), dim=0)

        try:
            aframes = aframes_to_fbank(aframes, info['audio_fps'], self.num_mel_bins, self.audio_target_length)
        except AssertionError:
            print(f"aframes_to_fbank failed for video: {filename}")
            print(info)
            print(aframes.shape)
            print(f"{clip_t_start}, {clip_t_end}")
        # TODO: Normalization with dataset mean & stddev?
        sample['clip'] = {
          "video": self.video_transform(vframes),
          "audio": aframes
        }
        sample['filename'] = filename
        sample['is-last-clip'] = is_last_clip

        return sample

    def save_output(self, batch_output, batch_input, label_columns):
        batch_output = [x.detach().cpu().numpy() for x in batch_output]

        for i in range(batch_output[0].shape[0]):
            filename, is_last_clip = batch_input['filename'][i], batch_input['is-last-clip'][i]
            if not (filename in self.saved_results):
                self.saved_results[filename] = {l: [] for l in label_columns}
            for j, label in enumerate(label_columns):
                self.saved_results[filename][label].append(batch_output[j][i,...])

            if is_last_clip:
                # dump results in disk at self.output_dir and then remove from self.saved_results
                output_filename = os.path.join(self.output_dir, os.path.basename(filename).split('.')[0] + '.pkl')
                for label in label_columns:
                    self.saved_results[filename][label] = np.stack(self.saved_results[filename][label])
                with open(output_filename, 'wb') as fobj:
                    pkl.dump(self.saved_results[filename], fobj)
                del self.saved_results[filename]

    def save_features(self, batch_features, batch_input):
        batch_features = batch_features.detach().cpu().numpy()

        for i in range(batch_features.shape[0]):
            filename, is_last_clip = batch_input['filename'][i], batch_input['is-last-clip'][i]
            if not (filename in self.saved_features):
                self.saved_features[filename] = []
            self.saved_features[filename].append(batch_features[i,...])

            if is_last_clip:
                # dump features to disk at self.output_dir and remove them from self.saved_features
                output_filename = os.path.join(self.output_dir, os.path.basename(filename).split('.')[0] + '.pkl')
                self.saved_features[filename] = np.stack(self.saved_features[filename])
                with open(output_filename, 'wb') as fobj:
                    pkl.dump(self.saved_features[filename], fobj)
                del self.saved_features[filename]
                print(f"Video features saved: {filename}")


    @staticmethod
    def _remove_unavailable_raw_videos_from_df(df: pd.DataFrame, root_dir, unavailable_videos):
        # get all available videos from root_dir
        videos = os.listdir(root_dir)

        # remove unavailable videos from dataframe
        df = df.loc[df['filename'].isin(videos)].copy()
        df = df.loc[~df['filename'].isin(unavailable_videos)]

        print(f"Number of available videos: {df.shape[0]}")

        return df

    
    @staticmethod
    def _make_filenames_absolute(df: pd.DataFrame, root_dir):
        # Change filenames in df to absolute filenames
        df['filename']= df['filename'].map(lambda f: os.path.join(root_dir, f))

        return df

    
    def _check_files_exist(df: pd.DataFrame):
        filenames = df.drop_duplicates('filename')['filename'].values
        for f in filenames:
            try:
                if not os.path.exists(f):
                    raise ValueError
            except ValueError:
                print(f'[EvalVideoDataset]: file {f} does not exist. '
                            f'Double-check root_dir and csv_filename inputs.')
                pass



    @staticmethod
    def _generate_clips_metadata(df, clip_length, frame_rate, stride):
        clip_metadata = {
            'filename': [],
            'fps': [],
            'clip-t-start': [],
            'is-last-clip': [],
        }
        for _, row in df.iterrows():
            total_frames_after_resampling = int(row['video-frames'] * (float(frame_rate) / row['fps']))
            idxs = EvalVideoDataset._resample_video_idx(total_frames_after_resampling, row['fps'], frame_rate)
            if isinstance(idxs, slice):
                frame_idxs = np.arange(row['video-frames'])[idxs]
            else:
                frame_idxs = idxs.numpy()
            clip_t_start = list(frame_idxs[np.arange(0,frame_idxs.shape[0]-clip_length+1,stride)]/row['fps'])
            num_clips = len(clip_t_start)

            clip_metadata['filename'].extend([row['filename']]*num_clips)
            clip_metadata['fps'].extend([row['fps']]*num_clips)
            clip_metadata['clip-t-start'].extend(clip_t_start)
            is_last_clip = [0] * num_clips
            is_last_clip[-1] = 1
            clip_metadata['is-last-clip'].extend(is_last_clip)

        return pd.DataFrame(clip_metadata)

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs