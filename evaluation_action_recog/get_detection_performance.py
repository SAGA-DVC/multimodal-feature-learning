import argparse
import numpy as np

from .eval_detection import ANETdetection

def run_eval(eval_args, ground_truth_json, prediction_json,
         subset='validation', check_status=True):

    tiou_thresholds = eval_args.tiou_thresholds
    verbose = eval_args.verbose
    
    anet_detection = ANETdetection(ground_truth_json, prediction_json,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=True, is_submission_json=True)
    mAP = anet_detection.evaluate()
    mAP_dict = {f'tiou-{tiou_threshold}':mAP_val for tiou_threshold, mAP_val in zip(tiou_thresholds, mAP)}
    return mAP_dict

def main(ground_truth_filename, prediction_filename,
         subset='validation', tiou_thresholds=np.linspace(0.5, 0.95, 10),
         verbose=True, check_status=True):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=True)
    anet_detection.evaluate()

def parse_input():
    description = ('This script allows you to evaluate the ActivityNet '
                   'detection task which is intended to evaluate the ability '
                   'of  algorithms to temporally localize activities in '
                   'untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-g', '--ground_truth_filename', type=str, default='sample_val.json',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('-p', '--prediction_filename', type=str, default='uniform_random_proposals.json',
                   help='Full path to json file containing the predictions.')
    p.add_argument('--subset', default='validation',
                   help=('String indicating subset to evaluate: '
                         '(training, validation)'))
    p.add_argument('--tiou_thresholds', type=float, default=np.linspace(0.5, 0.95, 10),
                   help='Temporal intersection over union threshold.')
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--check_status', type=bool, default=True)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_input()
    main(**vars(args))