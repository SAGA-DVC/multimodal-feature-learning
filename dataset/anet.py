import json
import h5py
from pathlib import Path
from collections import Counter, defaultdict
from itertools import chain
import pickle

import numpy as np
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

from config.config_dvc import load_config


class DVCdataset(Dataset):

    def __init__(self, annotation_file, feature_folder, tokenizer, vocab, is_training, args):

        """
        Parent class of the dataset to be used for training the DVC model (activity-net)

        Parameters:
            `annotation_file` (string) : path to json file consisting of a an object with the following format 
                                {video_id : {"duration": 82.73, 
                                            "timestamps": [[start_time_1, end_time_1], [start_time_2, end_time_2], [start_time_3, end_time_3]], 
                                            "sentences": [sentence_1, sentence_2, sentence_3],
                                            "action_labels" : [label_1, label_2, label_3]
                                            }
                                        }
            `feature_folder` (string) : Path to the file consisting of the features of all videos in the dataset
            `tokenizer` : Spacy english tokenizer for captions
            `vocab` (torchtext.vocab.Vocab) : mapping of all the words in the training dataset to indices and vice versa
            `is_training` (boolean) : 
            `args` (ml_collections.ConfigDict): Configuration object for the dataset
        """

        super(DVCdataset, self).__init__()

        self.annotation = json.load(open(annotation_file, 'r'))

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.PAD_IDX = vocab['<pad>']
        self.BOS_IDX = vocab['<bos>']
        self.EOS_IDX = vocab['<eos>']
        self.UNK_IDX = vocab['<unk>']
        self.max_caption_len_all = args.max_caption_len_all

        self.keys = list(self.annotation.keys())

        if args.invalid_videos_json is not None:
            invalid_videos = json.load(open(args.invalid_videos_json))
            self.keys = [k for k in self.keys if k not in invalid_videos]

        print(f'{len(self.keys)} videos are present in the dataset.')

        # for testing purposes (remove later)
        self.keys = self.keys[:12]

        # self.feature_folder = feature_folder
        self.video_features = h5py.File(feature_folder / 'video_features.h5', 'r')    # h5 object with {key (video id) as string : value (num_tokens, d_model) as dataset object}

        self.feature_sample_rate = args.feature_sample_rate
        self.is_training = is_training
        self.max_gt_target_segments = args.max_gt_target_segments
        self.num_queries = args.num_queries
        self.args = args

    def __len__(self):

        """
        Returns : length of the dataset i.e number of videos ids (or keys) in the dataset
        """

        return len(self.keys)


    def process_time_step(self, duration, timestamps_list, num_frames):

        """
        Converts the timestamps i.e (start_time, end_time) to (start_frame, end_frame). 
        
        Parameters:
            `duration` (float) : total duration of the video in seconds
            `timestamps_list` : 2d list of floats [gt_target_segments, 2], representing the start and end times of different events in the video
            `num_frames` (int) : number of frames in the video

        Returns:
            `framestamps` : 2d list of ints [gt_target_segments, 2], representing the start and end frames of different events in the video
        """

        duration = np.array(duration)
        timestamps = np.array(timestamps_list)
        num_frames = np.array(num_frames)
        framestamps = num_frames * timestamps / duration
        framestamps = np.minimum(framestamps, num_frames - 1).astype('int')
        framestamps = np.maximum(framestamps, 0).astype('int')
        return framestamps.tolist()

    def __getitem__(self, idx):
        raise NotImplementedError()


class ActivityNet(DVCdataset):

    def __init__(self, annotation_file, feature_folder, tokenizer, vocab, is_training, args):

        """
        Class for the ActivityNet dataset to be used for Dense Video Caption 

        Parameters:
            `annotation_file` (string) : path to json file consisting of a an object with the following format 
                                {video_id : {"duration": total_time, 
                                            "timestamps": [[start_time_1, end_time_1], [start_time_2, end_time_2], [start_time_3, end_time_3]], 
                                            "sentences": [sentence_1, sentence_2, sentence_3],
                                            "action_labels" : [label_1, label_2, label_3]
                                            }
                                        }
            `feature_folder` (string) : Path to the file consisting of the features of all videos in the dataset
            `tokenizer` : Spacy english tokenizer for captions
            `vocab` (torchtext.vocab.Vocab) : mapping of all the words in the training dataset to indices and vice versa
            `is_training` (boolean) :  
            `args` (ml_collections.ConfigDict): Configuration object for the dataset
        """

        super(ActivityNet, self).__init__(annotation_file, feature_folder, tokenizer, vocab, is_training, args)


    def load_feature(self, key):  

        """
        Loads the video features for a specific video and resizes it based on the \
        rescale length or sample rate (mentioned in the config object -- selg.args)

        Parameters:
            `key` (string) : An unique string representing a video in the dataset.

        Returns:
            `feature` : Tensor of dimension (num_tokens, d_model) 
        """

        feature = self.get_feature(key) # (num_tokens, d_model)
        
        if self.args.data_rescale == 'interpolate':
            feature = self.resizeFeature(feature, self.args.rescale_len) # (rescale_len, d_model)
        elif self.args.data_rescale == 'uniform':
            feature = feature[::self.args.feature_sample_rate] # (num_tokens//feature_sample_rate, d_model)

        return torch.from_numpy(feature)


    def __getitem__(self, idx):

        """
        Returns a single sample from the ActivityNet dataset

        Parameters:
            `idx` (int) : An ID representing a video in the dataset.

        Returns:
            `feature` (tensor): Tensor of dimension (num_tokens, d_model) representing the video
            `gt_framestamps` (list): 2d list of ints, [gt_target_segments, 2], representing the start and end frames of the events in the video 
            `action_labels` (list): [gt_target_segments] representing class labels for each event in the video 
            `captions_label` (list): [gt_target_segments, max_caption_len] {variable dim 1} representing the token labels for the caption of each event in the video
            `gt_timestamps` (list): 2d list of floats, [gt_target_segments, 2], representing the start and end times of the events in the video  
            `duration` (float) : representing the duration/length of the entire video in seconds
            `captions` (list): [gt_target_segments] consisting of the  caption of each event in the video
            `key` (string) : An unique string representing a video in the dataset.
        """

        key = self.keys[idx] # string

        feature = self.load_feature(key) # (num_tokens, d_model) 

        duration = self.annotation[key]['duration'] # float
        captions = self.annotation[key]['sentences'] # [gt_target_segments] -- list of strings
        gt_timestamps = self.annotation[key]['timestamps']  # [gt_target_segments, 2] -- 2d list of floats

        action_labels = self.annotation[key].get('action_labels', [0] * len(gt_timestamps)) # [gt_target_segments] -- default value [0, 0, 0...]
        assert max(action_labels) <= self.args.num_classes, f'action label {max(action_labels)} for video {key} is > total number of classes {self.args.num_classes}.'

        gt_target_segments = len(gt_timestamps) if len(gt_timestamps) < self.max_gt_target_segments else self.max_gt_target_segments
        random_gt_proposal_ids = np.random.choice(list(range(len(gt_timestamps))), gt_target_segments, replace=False) # (gt_target_segments) -- list

        captions = [captions[_] for _ in range(len(captions)) if _ in random_gt_proposal_ids] # [gt_target_segments] -- list of strings
        gt_timestamps = [gt_timestamps[_] for _ in range(len(gt_timestamps)) if _ in random_gt_proposal_ids] # [gt_target_segments, 2] -- 2d list of floats
        action_labels = [action_labels[_] for _ in range(len(action_labels)) if _ in random_gt_proposal_ids] # [gt_target_segments] -- default value [0, 0, 0...]

        captions_label = []    # [gt_target_segments, max_caption_len_all] {variable dim 1}

        for caption in captions:
            caption_label = [self.vocab[token] if token in self.vocab.get_itos() else self.UNK_IDX for token in self.tokenizer(caption)]
            caption_label = [self.BOS_IDX] + caption_label[:self.max_caption_len_all - 2] + [self.EOS_IDX]
            captions_label.append(caption_label)

        gt_framestamps = self.process_time_step(duration, gt_timestamps, feature.shape[0]) # [gt_target_segments, 2] -- 2d list of ints

        return feature, gt_framestamps, action_labels, captions_label, gt_timestamps, duration, captions, key


    def get_feature(self, key):

        """
        Extracts RGB features from a video

        Parameters:
            `key` (string) : An ID representing a video in the dataset
        
        Returns:
            np array of shape (num_tokens, d_model)
        """

        feature = np.array(self.video_features.get(key)).astype(np.float)    # (num_tokens, d_model)
        return feature


    def resizeFeature(self, input_data, new_size):

        """
        Resizes the video (number of frames) using interpolation

        Parameters:
            `input_data` : np array of shape (num_tokens, d_model)
            `new_size` (int) : num_tokens to be scaled to new_size

        Returns:
            `y_new` : np array of shape (new_size, d_model)
        """

        originalSize = len(input_data)

        if originalSize == 1:
            input_data = np.reshape(input_data, [-1])
            return np.stack([input_data] * new_size)
        
        x = np.array(range(originalSize))
        f = interp1d(x, input_data, axis=0, kind='nearest')
        x_new = [i * float(originalSize - 1) / (new_size - 1) for i in range(new_size)]
        y_new = f(x_new)

        return y_new


def iou(interval_1, interval_2):
    interval_1, interval_2 = map(np.array, (interval_1, interval_2))
    start, end = interval_2[None, :, 0], interval_2[None, :, 1]
    start_i, end_i = interval_1[:, None, 0], interval_1[:, None, 1]
    intersection = np.minimum(end, end_i) - np.maximum(start, start_i)
    union = np.minimum(np.maximum(end, end_i) - np.minimum(start, start_i), end - start + end_i - start_i)
    iou = intersection.clip(0) / (union + 1e-8)
    return iou


def sort_events(proposal_data):
    for vid in proposal_data.keys():
        v_data = proposal_data[vid]
        v_data = [p for p in v_data if p['score'] > 0]
        tmp = sorted(v_data, key=lambda x: x['segment'])
        proposal_data[vid] = tmp
    return proposal_data


# TODO - extra loss for framestamps?
def collate_fn(batch, pad_idx):
    """
    Parameters:
        `batch` : list of shape (batch_size, 8) {8 attributes}
        `pad_idx` : index of the '<pad>' token in the vocabulary
    Attributes
        `feature_list` (tensor): Tensor of dimension (batch_size, num_tokens, d_model) representing the video
        `gt_timestamps_list` (list, int): [batch_size, gt_target_segments, 2], representing the start and end frames of the events in the video 
        `action_labels` (list, int): [batch_size, gt_target_segments] representing class labels for each event in the video 
        `caption_list` (list, int): [batch_size, gt_target_segments, max_caption_len] {variable dim 1 and 2} representing the token labels for the caption of each event in the video
        `gt_raw_timestamps` (list, float): [batch_size, gt_target_segments, 2], representing the start and end times of the events in the video  
        `raw_duration` (list, float) : [batch_size], representing the duration/length of the entire video in seconds
        `raw_caption` (list, string): [batch_size, gt_target_segments] consisting of the  caption of each event in the video
        `key` (list, string) : [batch_size], An unique string representing a video in the dataset.
    
    Returns:
        `obj` : 
    """

    batch_size = len(batch)
    _, d_model = batch[0][0].shape
    
    feature_list, gt_timestamps_list, action_labels, caption_list, gt_raw_timestamps, raw_duration, raw_caption, key = zip(*batch)

    max_video_length = max([x.shape[0] for x in feature_list]) 
    max_caption_len = max(chain(*[[len(caption) for caption in captions] for captions in caption_list]))
    total_caption_num = sum([len(captions) for captions in caption_list])

    gt_timestamps = list(chain(*gt_timestamps_list)) # (batch_size * gt_target_segments, 2) {dim 0 is an avg value}

    video_tensor = torch.FloatTensor(batch_size, max_video_length, d_model).zero_()
    video_length = torch.FloatTensor(batch_size, 3).zero_()  # num_frames, duration, gt_target_segments
    video_mask = torch.BoolTensor(batch_size, max_video_length).zero_()

    caption_tensor_all = torch.LongTensor(total_caption_num, max_caption_len).fill_(pad_idx)
    caption_length_all = torch.LongTensor(total_caption_num).zero_()
    caption_mask_all = torch.BoolTensor(total_caption_num, max_caption_len).zero_()
    caption_gather_idx = torch.LongTensor(total_caption_num).zero_()

    max_gt_target_segments = max(len(x) for x in caption_list)

    gt_segments_tensor = torch.zeros(batch_size, max_gt_target_segments, 2)

    total_caption_idx = 0

    for idx in range(batch_size):
        video_len = feature_list[idx].shape[0]

        gt_segment_length = len(gt_timestamps_list[idx])

        video_tensor[idx, :video_len] = feature_list[idx]
        video_length[idx, 0] = float(video_len)
        video_length[idx, 1] = raw_duration[idx]
        video_length[idx, 2] = gt_segment_length
        video_mask[idx, :video_len] = True

        caption_gather_idx[total_caption_idx:total_caption_idx + gt_segment_length] = idx

        gt_segments_tensor[idx, :gt_segment_length] = torch.tensor(
            [[(ts[1] + ts[0]) / (2 * raw_duration[idx]), (ts[1] - ts[0]) / raw_duration[idx]] for ts in
             gt_raw_timestamps[idx]]).float()

        for iidx, caption in enumerate(caption_list[idx]):
            _caption_len = len(caption)
            caption_length_all[total_caption_idx + iidx] = _caption_len
            caption_tensor_all[total_caption_idx + iidx, :_caption_len] = torch.Tensor(caption)
            caption_mask_all[total_caption_idx + iidx, :_caption_len] = 1
        
        # mask = (caption_length_all == self.PAD_IDX)
        # print(mask.shape)

        total_caption_idx += gt_segment_length

    gt_segments_mask = (gt_segments_tensor != 0).sum(2) > 0    # (batch_size, max_gt_target_segments)

    target = [{'segments': torch.tensor([[(ts[1] + ts[0]) / (2 * raw_duration[i]), 
                            (ts[1] - ts[0]) / raw_duration[i]] 
                            for ts in gt_raw_timestamps[i]]).float(),    # (gt_target_segments, 2)
               'labels': torch.tensor(action_labels[i]).long(),    # (gt_target_segments)
               'masks': None,
               'vid_id': vid} for i, vid in enumerate(list(key))]

    obj = {
        "video":
            {
                "tensor": video_tensor,    # (batch_size, max_video_length, d_model)
                "length": video_length,    # (batch_size, 3) - num_frames, duration, gt_target_segments
                "mask": video_mask,    # (batch_size, max_video_length)
                "key": list(key),    # list, (batch_size)
                "target": target,
            },
       
        "gt":
            {
                "framestamps": gt_timestamps,  # list, (batch_size * gt_target_segments, 2) {dim 0 is an avg value}
                "timestamp": list(gt_raw_timestamps),  # list of tensors, (shape: (batch_size, gt_target_segments, 2) {variable dim 1})
                "gather_idx": caption_gather_idx,  # tensor, (total_caption_num)
                "segments": gt_segments_tensor, # (batch_size, max_gt_target_segments, 2)
                "segments_mask": gt_segments_mask, # (batch_size, max_gt_target_segments)
            },

        "cap":
            {
                "tensor": caption_tensor_all,  # tensor, (total_caption_num, max_caption_len)
                "length": caption_length_all,  # tensor, (total_caption_num)
                "mask": caption_mask_all,  # tensor, (total_caption_num, max_caption_len)
                "raw": list(raw_caption),  # list of strings, (batch_size, gt_target_segments) {variable dim 1}
            }
    }
    obj = {k1 + '_' + k2: v2 for k1, v1 in obj.items() for k2, v2 in v1.items()}
    return obj


def build_vocab(annotation, tokenizer):
        """
        Builds the vocabulary (word to idx and idx to word mapping) based on all the captions in the training dataset.
        """
        
        counter = Counter()

        captions = []
        for value in list(annotation.values()):
            captions += value['sentences']

        for caption in captions:
            counter.update(tokenizer(caption))

        return vocab(counter, min_freq=2, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def build_dataset(video_set, args):

    """
    Builds the ActivityNet dataset.

    Parameters:
        `video_set` (string) : Can be one of 'train' or 'val'. Used to build dataset for training or validation
        `args` (ml_collections.ConfigDict) :  Configuration object for the dataset
    """
    
    root_annotation = Path(args.anet_path)
    root_feature = Path(args.features_path)

    assert root_annotation.exists(), f'Provided ActivityNet path {root_annotation} does not exist.'
    assert root_feature.exists(), f'Provided ActivityNet feature folder path {root_feature} does not exist.'
    
    assert video_set in ['train', 'val'], f'video_set is {video_set} but should be one of "train" or "val".'

    PATHS_ANNOTATION = {
        "train": (root_annotation / 'train.json'),
        "val": (root_annotation / 'val_1.json'),
    }
    PATHS_VIDEO = {
        "train": (root_feature / 'train'),
        "val": (root_feature / 'val'),
    }

    annotation_file = PATHS_ANNOTATION[video_set]
    feature_folder = PATHS_VIDEO[video_set]
    
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    # TODO - save words in a diff file for faster vocab building
    vocab_file = Path(args.vocab_file_path)

    if vocab_file.exists():
        vocab = pickle.load(open(vocab_file, 'rb'))
    else:
        vocab = build_vocab(json.load(open("../activity-net/captions/train.json", 'r')), tokenizer)
        pickle.dump(vocab, open(vocab_file, 'wb'))
    

    dataset = ActivityNet(annotation_file=annotation_file, 
                          feature_folder=feature_folder,
                          tokenizer=tokenizer,
                          vocab=vocab,
                          is_training=(video_set == 'train'),
                          args=args)
    return dataset


if __name__ == '__main__':
    args = load_config()
    dataset_train = build_dataset(video_set='train', args=args.dataset.activity_net)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers)

    for obj in data_loader_train:
        print(obj.keys())

