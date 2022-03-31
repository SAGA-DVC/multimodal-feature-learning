'''
Code adapted from https://github.com/HumamAlwassel/TSP
Alwassel, H., Giancola, S., & Ghanem, B. (2021). TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
'''


import os
import sys
sys.path.insert(0, '..')
from pprint import pprint

import pandas as pd
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision.io import read_video

from models.ast_utils import aframes_to_fbank


class UntrimmedVideoDataset(Dataset):
    '''
    UntrimmedVideoDataset:
        This dataset takes in temporal segments from untrimmed videos and samples fixed-length
        clips from each segment. Each item in the dataset is a dictionary with the keys:
            - "clip": A Tensor (dtype=torch.float) of the clip frames after applying transforms
            - "label-Y": A label from the `label_columns` (one key for each label) or -1 if label is missing for that clip
            - "gvf": The global video feature (GVF) vector if `global_video_features` parameter is not None
    '''

    def __init__(self, csv_filename, root_dir, video_clip_length_frames, video_frame_rate, audio_frame_rate, clips_per_segment, temporal_jittering,
            label_columns, label_mappings, num_mel_bins, audio_target_length, video_h5=None, audio_h5=None, seed=42, video_transform=None, global_video_features=None, debug=False):
        '''
        Args:
            csv_filename (string): Path to the CSV file with temporal segments information and annotations.
                The CSV file must include the columns [filename, fps, t-start, t-end, video-duration] and
                the label columns given by the parameter `label_columns`.
            root_dir (string): Directory with all the raw video files. Specify either this or `video_h5` and `audio_h5`.
            clip_length (int): The number of frames per clip.
            video_frame_rate (int): The effective video frame rate (fps) to sample clips.
            audio_frame_rate (int): The effective audio frame rate (fps) to sample clips.
            clips_per_segment (int): The number of clips to sample per segment (a row) in the CSV file.
            temporal_jittering (bool): If True, clips are randomly sampled between t-start and t-end of
                each segment. Otherwise, clips are are sampled uniformly between t-start and t-end.
            num_mel_bins (int) TODO
            audio_target_length TODO
            video_h5 (string): Path to the HDF5 file of preprocessed video tensors
            audio_h5 (string): Path to the HDF5 file of preprocessed audio tensors
            seed (int): Seed of the random number generator used for the temporal jittering.
            video_transform (callable): A function/transform that takes in a TxHxWxC video
                and returns a transformed video.
            label_columns (list of string): A list of the label columns in the CSV file.
                If more than one column is specified, the sample return a label for each.
            label_mappings (list of dict): A list of dictionaries to map the corresponding label
                from `label_columns` from a category string to an integer ID value.
            global_video_features (string): Path to h5 file containing global video features (optional)
            debug (bool): If true, create a debug dataset with 100 samples.
        '''
        df = UntrimmedVideoDataset._clean_df_and_remove_short_segments(pd.read_csv(csv_filename), video_clip_length_frames, video_frame_rate)

        self.root_dir = root_dir

        self.video_h5 = video_h5
        self.audio_h5 = audio_h5

        if root_dir:
            # self.df = UntrimmedVideoDataset._append_root_dir_to_filenames_and_check_files_exist(df, root_dir)
            df = UntrimmedVideoDataset._remove_unavailable_raw_videos_from_df(df, root_dir)
            self.df = UntrimmedVideoDataset.make_filenames_absolute(df, root_dir)
            UntrimmedVideoDataset._check_files_exist(self.df)
        elif video_h5 and audio_h5:
            self.video_h5 = h5py.File(video_h5, "r")
            self.audio_h5 = h5py.File(audio_h5, "r")
            self.df = UntrimmedVideoDataset._remove_unavailable_tensors_from_df(df, self.video_h5, self.audio_h5)
        else:
            raise NotImplementedError

        self.video_clip_length_frames = video_clip_length_frames
        
        self.video_frame_rate = video_frame_rate
        self.audio_frame_rate = audio_frame_rate

        self.clip_length_sec = self.video_clip_length_frames / self.video_frame_rate

        self.audio_clip_length_frames = self.clip_length_sec * self.audio_frame_rate

        self.clips_per_segment = clips_per_segment

        self.audio_target_length = audio_target_length
        self.num_mel_bins = num_mel_bins

        self.temporal_jittering = temporal_jittering
        
        '''Used for temporal jittering, samples a number from [0, 1) uniformly'''
        self.rng = np.random.RandomState(seed=seed)

        '''Used for uniform sampling of indices of clips in a segment'''
        self.uniform_sampling = np.linspace(0, 1, clips_per_segment)

        self.video_transform = video_transform

        self.label_columns = label_columns
        self.label_mappings = label_mappings
        for label_column, label_mapping in zip(label_columns, label_mappings):
            self.df[label_column] = self.df[label_column].map(lambda x: -1 if pd.isnull(x) else label_mapping[x])

        self.global_video_features = global_video_features
        if global_video_features:
            self.gvf_h5 = h5py.File(self.global_video_features, 'r')

        self.debug = debug

    def __del__(self):
        if self.global_video_features:
            self.gvf_h5.close()
        if self.video_h5:
            self.video_h5.close()
        if self.audio_h5:
            self.audio_h5.close()

    def __len__(self):
        num_clips = len(self.df) * self.clips_per_segment
        return num_clips if not self.debug else min(num_clips, 100)

    def __getitem__(self, idx):
        sample = {}

        row = self.df.iloc[idx % len(self.df)]
        video_id, filename, fps, segment_start_sec, segment_end_sec = row['video-name'], row['filename'], row['fps'], row['t-start'], row['t-end']

        # For calculating the start of clip
        ratio = self.rng.uniform() if self.temporal_jittering else self.uniform_sampling[idx//len(self.df)]

        if self.root_dir:
            # Use raw video

            clip_start_sec = segment_start_sec + ratio * (segment_end_sec - segment_start_sec - self.clip_length_sec)
            clip_end_sec = clip_start_sec + self.clip_length_sec

            # get a video tensor [clip_length, H, W, C] and audio tensor [channels, points]
            # of the video frames between clip_t_start and clip_t_end seconds
            vframes, aframes, info = read_video(filename=filename, start_pts=clip_start_sec, end_pts=clip_end_sec, pts_unit='sec')

        elif self.video_h5 and self.audio_h5:
            # Use precomputed video and audio tensors
            
            v_segment_start_frame = fps * segment_start_sec
            v_segment_end_frame = fps * segment_end_sec

            v_clip_start_frame = int(v_segment_start_frame + ratio * (v_segment_end_frame - v_segment_start_frame - self.video_clip_length_frames))
            v_clip_end_frame = v_clip_start_frame + self.video_clip_length_frames

            vframes = torch.tensor(self.video_h5[video_id][:, v_clip_start_frame:v_clip_end_frame, :, :])
            
            # df does not have audio frame rate
            a_segment_start_frame = self.audio_frame_rate * segment_start_sec
            a_segment_end_frame = self.audio_frame_rate * segment_end_sec

            a_clip_start_frame = int(a_segment_end_frame + ratio * (a_segment_end_frame - a_segment_start_frame - self.audio_target_length))
            a_clip_end_frame = a_clip_start_frame + self.audio_clip_length_frames
            
            aframes = torch.tensor(self.audio_h5[video_id][0][a_clip_start_frame:a_clip_end_frame])

        # If video has different FPS than self.frame_rate, then change idxs to reflect this
        # If video fps == self.frame_rate, then idxs is simply [::1]
        idxs = UntrimmedVideoDataset._resample_video_idx(self.video_clip_length_frames, fps, self.video_frame_rate)

        # TODO resampling index for audio frames
        # As of now, all videos are assumed to have same audio_frame_rate (audio sample rate)
        # This is true for almost all videos except ~10-20
        # But the audio sample rate is not stored in df

        vframes = vframes[idxs][:self.video_clip_length_frames]  # [:self.clip_length] for removing extra frames if isinstance(idxs, slice)

        if vframes.shape[0] != self.video_clip_length_frames:
            raise RuntimeError(f'<UntrimmedVideoDataset>: got clip of length {vframes.shape[0]} != {self.video_clip_length_frames}.'
                               f'filename={filename}, clip_t_start={clip_start_sec}, clip_t_end={clip_end_sec}, '
                               f'fps={fps}, t_start={segment_start_sec}, t_end={segment_end_sec}')

        # apply video transforms
        sample['video'] = self.video_transform(vframes)

        # apply audio transforms
        aframes = aframes_to_fbank(aframes, info['audio_fps'], self.num_mel_bins, self.audio_target_length)
        # TODO: Spectogram Augmentation: Frequency Masking, Time Masking (only for training set)
        # TODO: Normalization with dataset mean & stddev?
        sample['audio'] = aframes

        # add labels
        for label_column in self.label_columns:
            sample[label_column] = row[label_column]

        # add global video feature
        if self.global_video_features:
            sample['gvf'] = torch.tensor(self.gvf_h5[os.path.basename(filename).split('.')[0]][()])

        return sample

    @staticmethod
    def _clean_df_and_remove_short_segments(df, clip_length, frame_rate):
        # restrict all segments to be between [0, video-duration]
        df['t-end'] = np.minimum(df['t-end'], df['video-duration'])
        df['t-start'] = np.maximum(df['t-start'], 0)

        # remove segments that are too short to fit at least one clip
        segment_length = (df['t-end'] - df['t-start']) * frame_rate
        mask = segment_length >= clip_length
        num_segments = len(df)
        num_segments_to_keep = sum(mask)

        if num_segments - num_segments_to_keep > 0:
            # if at least one segment is applicable
            df = df[mask].reset_index(drop=True)
            print(f'<UntrimmedVideoDataset>: removed {num_segments - num_segments_to_keep}='
                f'{100*(1 - num_segments_to_keep/num_segments):.2f}% from the {num_segments} '
                f'segments from the input CSV file because they are shorter than '
                f'clip_length={clip_length} frames using frame_rate={frame_rate} fps.')
        else:
            raise NotImplementedError

        return df

    @staticmethod
    def _remove_unavailable_raw_videos_from_df(df: pd.DataFrame, root_dir):
        # get all available videos from root_dir
        video_filenames = os.listdir(root_dir)

        # remove unavailable videos from dataframe
        df = df.loc[df['filename'].isin(video_filenames)].copy()
        
        print(f"Number of segments after removing unavailable raw videos: {df.shape[0]}")

        return df

    @staticmethod
    def _remove_unavailable_tensors_from_df(df, video_h5, audio_h5):
        videos = set(video_h5.keys())
        audios = set(audio_h5.keys())
        if len(audios) != len(videos) or audios != videos:
            print("Audio datasets in audio_h5 don't match with video datasets in video_h5")
        ids = videos.intersection(audios)
        
        # remove unavailable videos from dataframe
        df = df.loc[df['video-name'].isin(ids)].copy()

        return df


    @staticmethod
    def make_filenames_absolute(df, root_dir):
        # Change filenames in df to absolute filenames
        df['filename']= df['filename'].map(lambda f: os.path.join(root_dir, f))

        return df

    @staticmethod
    def _check_files_exist(df):
        filenames = df.drop_duplicates('filename')['filename'].values
        pprint(len(filenames))
        for f in filenames:
            try:
                if not os.path.exists(f):
                    raise ValueError(f'<UntrimmedVideoDataset>: file={f} does not exists. '
                                    f'Double-check root_dir and csv_filename inputs.')
            except ValueError:
                # print(f"Video {f} not present")
                pass

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
