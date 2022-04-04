import torch
import json


def get_sample_submission():
    return {
        "version": "VERSION 1.0",
        "results": {
            "v_uqiMw7tQ1Cc": [
                {
                    "sentence": "One player moves all around the net holding the ball",
                    "timestamp": [1.23, 4.53]
                },
                {
                    "sentence": "A small group of men are seen running around a basketball court playing a game",
                    "timestamp": [5.24, 18.23]
                }
            ]
        },
        "external_data": {
            "used": True, 
            "details": "PDVC + BMT + Event-Centric + Thoda Novelty"
        }
    }


def get_src_permutation_idx(indices):
    '''
    Parameters:
    `indices` (list): list (len=batch_size) of tuple of tensors (shape=(2, gt_target_segments))
                    eg. [ (tensor[2, 8, 61], tensor[2, 0, 1]), (...), ... ] --> segment 2, 8, 61 from video 1; Order = 8, 61, 2
    Returns:
    `batch_idx` (tensor): (nb_target_segments) contains batch number; eg. [0, 0, 0,   1, 1] 
    `src_idx` (tensor): (nb_target_segments) contains source indices of bipartite matcher; eg. [2, 14, 88,   3, 91]
    '''

    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([torch.tensor([src[i] for i in torch.sort(b)[1]], dtype=torch.int32) for (src, b) in indices])
    return batch_idx, src_idx


def denormalize_segments(segments, video_durations):
    '''
    Parameters:
    `segments` (tensor): (batch_size, num_proposals, 2), representing center offset and length offset
    `video_durations` (tensor, float): (batch_size,), representing duration of videos

    Returns:
    `denormalized_segments` (tensor):  (batch_size, num_proposals, 2), representing start_time and end_time
    '''
    
    batch_size = segments.shape[0]
    denormalized_segments = torch.zeros(list(segments.shape), dtype=torch.float32)
    print("Shapes: ", segments.shape, denormalized_segments.shape)

    for idx in range(batch_size):
        d = video_durations[idx]
        denormalized_segments[idx] = torch.tensor(
            [[(d/2 * (2*cl[0] - cl[1])), (d/2 * (2*cl[0] + cl[1]))] for cl in segments[idx]]
        ).float()

    return denormalized_segments


def captions_to_string():
    pass


def save_submission(json_data, json_file_path):
    with open(json_file_path, 'w') as f:
        json.dump(json_data, f)