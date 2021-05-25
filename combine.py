import os
import glob
import argparse
import cv2
import pickle
import math
import numpy as np
import util.percentiles

def build_args():
    parser = argparse.ArgumentParser(description='infernece.py')
    parser.add_argument('--input_dir', '-input_dir', type=str, required=True,
                        help='Data folder directory which contains error.<fold-ID>.pkl.')
    args = parser.parse_args()
    return args

def main(args):
    errors = []
    
    for path in glob.glob(os.path.join(args.input_dir, 'errors.*.pkl')):
        with open(path, 'rb') as f:
            errors += pickle.load(f)
        
    summary = util.percentiles.percentiles(errors)
    print(summary)
    
    with open(os.path.join(args.input_dir, "summary.pkl"), 'wb') as f:
        pickle.dump(summary, f)
        
if __name__ == '__main__':
    args = build_args()
    main(args)