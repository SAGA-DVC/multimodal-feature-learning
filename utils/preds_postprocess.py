import torch
import json


def get_sample_submission():
    return {
        "version": "VERSION 1.0",
        "results": {},
        "external_data": {
            "used": True, 
            "details": "DVC"
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
    src_idx = torch.cat([torch.tensor([src[i] for i in torch.sort(b)[1]], dtype=torch.long) for (src, b) in indices])
    return batch_idx, src_idx

'''
def denormalize_segments(segments, video_durations):
    # Parameters:
    # `segments` (tensor): (batch_size, num_proposals, 2), representing center offset and length offset
    # `video_durations` (tensor, float): (batch_size,), representing duration of videos

    # Returns:
    # `denormalized_segments` (tensor):  (batch_size, num_proposals, 2), representing start_time and end_time
    
    batch_size = segments.shape[0]
    denormalized_segments = torch.zeros(list(segments.shape), dtype=torch.float32)
    print("Shapes: ", segments.shape, denormalized_segments.shape)

    for idx in range(batch_size):
        d = video_durations[idx]
        denormalized_segments[idx] = torch.tensor(
            [[(d/2 * (2*cl[0] - cl[1])), (d/2 * (2*cl[0] + cl[1]))] for cl in segments[idx]]
        ).float()

    return denormalized_segments
'''


def denormalize_segments(segments, video_durations, segment_batch_id):
    '''
    Parameters:
    `segments` (tensor): (batch_size * num_proposals, 2), representing center offset and length offset
    `video_durations` (tensor, float): (batch_size,), representing duration of videos
    `segment_batch_id` (tensor, int): (num_proposals,), representing batch id of corresponding segment

    Returns:
    `denormalized_segments` (tensor):  (batch_size * num_proposals, 2), representing start_time and end_time
    '''
    
    # batch_size = segments.shape[0]
    denormalized_segments = torch.zeros(list(segments.shape), dtype=torch.float32)
    # print("Segments Shapes: ", segments.shape, denormalized_segments.shape, video_durations, segment_batch_id)

    for i, idx in enumerate(segment_batch_id):
        d = video_durations[idx]
        denormalized_segments[i] = torch.tensor(
            [min(max((d/2 * (2*segments[i][0] - segments[i][1])), 0), d), max(min((d/2 * (2*segments[i][0] + segments[i][1])), d), 0)]
        ).float()

    # TODO - Interchange if start > end
    denormalized_segments = torch.tensor(list(map(
        lambda p: p.tolist() if p[0] < p[1] else p.tolist()[::-1], denormalized_segments
    )))

    return denormalized_segments


def captions_to_string(captions, vocab):
    '''
    Convert captions (token) to string

    Parameters:
    `captions` (tensor): (total_caption_num, max_caption_length, vocab_size)
    `vocab` (torchtext.vocab.Vocab): mapping of all the words in the training dataset to indices and vice versa)
    '''
    
    PAD_IDX = vocab['<pad>']
    BOS_IDX = vocab['<bos>']
    EOS_IDX = vocab['<eos>']
    UNK_IDX = vocab['<unk>']
    unwanted_tokens = [PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX]

    captions_string = []
    for caption in captions:
        captions_string.append(' '.join([vocab.get_itos()[token_num] for token_num in caption if token_num not in unwanted_tokens][1:-1]))

    captions_string = pre_process(captions_string)

    return captions_string


def save_submission(json_data, json_file_path):
    with open(json_file_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    
def pprint_eval_scores(scores, debug=False):
    # Print the averages
    if debug:
        print ('-' * 80)
        print ("Average across all tIoUs")
        print ('-' * 80)
    
    avg_scores = {}
    for metric in scores:
        if type(scores[metric]) == list:
            score = scores[metric]
            avg_scores[metric] = 100 * sum(score) / float(len(score))
            
            if debug:
                print ('| %s: %2.4f'%(metric, avg_scores[metric]))
        else:
            avg_scores[metric] = scores[metric]
    
    if (avg_scores['Precision'] + avg_scores['Recall']) > 0:
        avg_scores['F1_score'] = 2 * (avg_scores['Precision'] * avg_scores['Recall']) / (avg_scores['Precision'] + avg_scores['Recall'])
    else:
        avg_scores['F1_score'] = -2 * (avg_scores['Precision'] * avg_scores['Recall'])
    
    return avg_scores


def pre_process(captions):
    for i, caption in enumerate(captions):
        tokens = caption.split()
        if len(tokens) == 0 :
            captions[i] = ''    # TODO check if space separated
            continue
        res_tokens = [tokens[0]]
        for j in range(1, len(tokens)):
            if tokens[j] in ['.', ',', '/', "'"]:
                continue
            else:
                if res_tokens[-1] == tokens[j] : continue
                else : res_tokens.append(tokens[j])
        captions[i] = ' '.join(res_tokens)
    return captions
    