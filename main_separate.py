from pathlib import Path

from config.config_dvc import load_config
from main_cap import train_cap
from main_prop import train_prop

if __name__ == '__main__':
    args = load_config()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.submission_dir:
        Path(args.submission_dir).mkdir(parents=True, exist_ok=True)
    
    if args.procedure == 'train_cap':
        train_cap(args)
    elif args.procedure == 'train_prop':
        train_prop(args)

    # elif args.procedure == 'evaluate':
    #     train_cap(args)
