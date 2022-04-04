import argparse

import torch
import h5py
from tqdm import tqdm

def main(args):
    print(args)
    compression_flags = dict(compression='gzip', compression_opts=9)

    # Open features h5 file
    with h5py.File(args.features_h5, 'r') as f:
        # Open output gvf h5 file
        with h5py.File(args.output_h5, 'a') as output:
            # For each video
            for (video_id, video_clips) in tqdm(f.items()):
                if f.get(video_id) is None:
                    if args.pooling_fn == 'max':
                        gvf, _ = torch.tensor(video_clips).max(dim=0)
                    elif args.pooling_fn == 'avg':
                        gvf, _ = torch.tensor(video_clips).mean(dim=0)

                    output.create_dataset(video_id, data=gvf, **compression_flags)

    print(f"Saved output file at {args.output_h5}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pool from clips of each video to get a single GVF')

    parser.add_argument('--features-h5', required=True, type=str,
                      help='Path to the h5 file having video clip features')
    parser.add_argument('--output-h5', required=True, type=str,
                      help='Where to save the GVF h5 file')

    parser.add_argument('--pooling-fn', type=str, default='max', help='Pooling function to use', choices=['avg', 'max'])
    args = parser.parse_args()

    main(args)