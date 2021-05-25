import os
import argparse
import cv2
import pickle
import math
import tensorflow as tf
import numpy as np
import util.percentiles
import model.fc4

from tqdm import tqdm
from collections import OrderedDict
from dataloader import Dataloader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow all warnings.

try: # Disable annoying info and warnings.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    tf.logging.set_verbosity(tf.logging.ERROR)
    
slim = tf.contrib.slim

def build_args():
    parser = argparse.ArgumentParser(description='infernece.py')
    parser.add_argument('--data_dir', '-data_dir', type=str, required=True,
                        help='Data folder directory which contains "gehler", "nus" folder.')
    parser.add_argument('--test_data', '-test_data', type=str, required=True,
                        help='["gehler", "nus"]')
    parser.add_argument('--test_fold', '-test_fold', type=str, required=True,
                        help='[0,1,2]')
    parser.add_argument('--backbone', '-backbone', type=str, required=True,
                        help='["alexnet", "squeezenet"]')
    parser.add_argument('--ckpt_path', '-ckpt_path', type=str, required=True,
                        help='Checkpoint path')
    parser.add_argument('--output_dir', '-output_dir', type=str, default='inference_results')
    args = parser.parse_args()
    return args

def main(args, sess):
    """ Build dataset """
    test_dataloader = Dataloader(args.data_dir, args.test_data, [args.test_fold], batch_size=1, is_training=False)

    """ Build model graph """
    x = {
        "images": tf.placeholder(tf.float32, shape=(None, None, None, 3), name='images'),
        "illums": tf.placeholder(tf.float32, shape=(None, 3), name='illums')
    }
    
    output = model.fc4.model(x["images"], is_training=False, reuse=tf.AUTO_REUSE, backbone_type=args.backbone)
        
    # Load ckpt
    print('INFO:Loading ckpt from "%s"...' % args.ckpt_path)
    vars_to_restore = tf.contrib.framework.get_variables_to_restore()
    restore_fn = slim.assign_from_checkpoint_fn(args.ckpt_path, vars_to_restore, ignore_missing_vars=True)
    restore_fn(sess)
    
    # Inference
    errors = []
    num_test_iterations = test_dataloader.data_count
    for _ in tqdm(range(num_test_iterations)):
        batch = sess.run(test_dataloader.get_batch)
        
        pred_illums = sess.run(output['illums'], feed_dict={
            x["images"]: batch[0],
            x["illums"]: batch[1]
        })

        pred_illum = pred_illums[0]
        gt_illum = batch[1][0]
        errors.append(util.percentiles.angular_error(pred_illum, gt_illum))
    
    # Save fold errors
    with open(os.path.join(args.output_dir, 'errors.%s.pkl' % args.test_fold), 'wb') as f:
        pickle.dump(errors, f)
        
    print(util.percentiles.percentiles(errors))
        
if __name__ == '__main__':
    args = build_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with tf.Session() as sess:
        main(args, sess)