import os
import argparse
import cv2
import pickle
import math
import tensorflow as tf
import numpy as np
import util.raw2raw
import util.perturb
import util.loss
import util.trigger
import util.percentiles
import model.fc4, model.mlp

from tqdm import tqdm
from collections import OrderedDict
from dataloader import Dataloader
from config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow all warnings.

try: # Disable annoying info and warnings.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    tf.logging.set_verbosity(tf.logging.ERROR)
    
slim = tf.contrib.slim

def print_error_percentile(exp_name, mode, cur_epoch, error_p):
    info = '[%s|%s] Epoch: %d' % (exp_name, mode, cur_epoch)

    for k in error_p.keys():
        info += '\n %s (mean): %5.3f' % (k, error_p[k]["mean"])

    print(info + '\n')
    
def main(sess):
    """ Build dataset """
    train_dataloader = Dataloader(DATA_DIR, DATA_NAME, TRAIN_FOLDS, batch_size=TRAIN_BATCH_SIZE, is_training=True)
    test_dataloader = Dataloader(DATA_DIR, DATA_NAME, TEST_FOLDS, batch_size=1, is_training=False)
    
    """ Build model graph """
    # Placeholders
    x = {
        "images": tf.placeholder(tf.float32, shape=(None, None, None, 3), name='images'),
        "illums": tf.placeholder(tf.float32, shape=(None, 3), name='illums'),
        "cc24s": tf.placeholder(tf.float32, shape=(None, 24, 3), name='cc24s')
    }
    
    # Augment raw-to-raw contrastive pairs
    a_images, \
    easy_p_images, hard_p_images,\
    easy_n_images, hard_n_images, \
    contrastive_loss_masks = util.raw2raw.construct_contrastive_pairs(x["images"], x["illums"], x["cc24s"], N_NEGATIVE)

    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(GLOBAL_WEIGHT_DECAY)):

        # Color constancy branch.
        train_out = model.fc4.model(x["images"], is_training=True, reuse=tf.AUTO_REUSE, backbone_type=BACKBONE)
        test_out = model.fc4.model(x["images"], is_training=False, reuse=tf.AUTO_REUSE, backbone_type=BACKBONE)

        # Contrastive learning branch.
        a_features = model.fc4.model(util.perturb.perturb(a_images),is_training=True, reuse=tf.AUTO_REUSE, backbone_type=BACKBONE)["features"]
        easy_p_features = model.fc4.model(util.perturb.perturb(easy_p_images), is_training=True, reuse=tf.AUTO_REUSE, backbone_type=BACKBONE)["features"]
        easy_n_features = model.fc4.model(util.perturb.perturb(easy_n_images), is_training=True, reuse=tf.AUTO_REUSE, backbone_type=BACKBONE)["features"]
        hard_p_features = model.fc4.model(util.perturb.perturb(hard_p_images), is_training=True, reuse=tf.AUTO_REUSE, backbone_type=BACKBONE)["features"]
        hard_n_features = model.fc4.model(util.perturb.perturb(hard_n_images), is_training=True, reuse=tf.AUTO_REUSE, backbone_type=BACKBONE)["features"]

        # Feature projection
        a_features = model.mlp.MLP(a_features, 'MLP', tf.AUTO_REUSE) # (b, c)
        easy_p_features = model.mlp.MLP(easy_p_features, 'MLP', tf.AUTO_REUSE)
        easy_n_features = model.mlp.MLP(easy_n_features, 'MLP', tf.AUTO_REUSE)
        hard_p_features = model.mlp.MLP(hard_p_features, 'MLP', tf.AUTO_REUSE)
        hard_n_features = model.mlp.MLP(hard_n_features, 'MLP', tf.AUTO_REUSE)
        
    """ Build loss """
    # Training loss
    color_constancy_loss = util.loss.angular_loss(train_out["illums"], x["illums"], average=True)
    reg_loss = tf.add_n(slim.losses.get_regularization_losses())

    contrastive_loss = 0
    for a, p, n in [(a_features, easy_p_features, easy_n_features),
                    (a_features, easy_p_features, hard_n_features),
                    (a_features, hard_p_features, easy_n_features),
                    (a_features, hard_p_features, hard_n_features)]:
        contrastive_losses = util.loss.nce_loss(a, p, n, N_NEGATIVE, average=False)
        contrastive_loss += tf.reduce_mean(contrastive_losses * contrastive_loss_masks) # for training.
    contrastive_loss /= 4

    # Testing loss
    test_loss = util.loss.angular_loss(test_out["illums"], x["illums"])

    """ Build training op """
    pretrain_loss = (0.1 * color_constancy_loss) + contrastive_loss + reg_loss
    pretrain_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(pretrain_loss)

    jointtrain_loss = color_constancy_loss + (0.1 * contrastive_loss) + reg_loss
    jointtrain_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(jointtrain_loss)
    
    """ Print variables """
    print('-'*100 + '\n')
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # All trainable vars.
    slim.model_analyzer.analyze_vars(var_list, print_info=True)
    print('-'*100 + '\n')
    
    """ Start training """
    trigger = util.trigger.Trigger(minimize=True)
    saver = tf.train.Saver(max_to_keep=None)
    num_train_iterations = train_dataloader.data_count // train_dataloader.batch_size # Iterations per training epoch.
    num_test_iterations = test_dataloader.data_count
    sess.run(tf.global_variables_initializer())
    
    # Squeezenet loads imagenet pretrained weights internally
    # Alexnet needs to load it explicitly.
    if BACKBONE == 'alexnet':
        # Variable scope should be consistent to current model graph.
        with tf.variable_scope('FCN/AlexNet', reuse=True):
            from model.alexnet import AlexNet
            AlexNet.load_initial_weights(sess)
            
    # Prevent memory leak.
    sess.graph.finalize()
    
    for i in range(1, PRETRAIN_EPOCHS+JOINTTRAIN_EPOCHS+1):
        if i <= PRETRAIN_EPOCHS:
            train_loss, train_op = pretrain_loss, pretrain_op
        else:
            train_loss, train_op = jointtrain_loss, jointtrain_op
            
        # Training. cc_loss (i.e., color constancy loss) is same as angular loss.
        train_errors = OrderedDict([('loss', []), ('cc_loss', []), ('cl_loss', [])])
        for _ in tqdm(range(num_train_iterations)):
            batch = sess.run(train_dataloader.get_batch)
            _, train_loss_val, color_constancy_loss_val, contrastive_loss_val = \
                sess.run([train_op, train_loss, color_constancy_loss, contrastive_loss], feed_dict={
                    x["images"]: batch[0],
                    x["illums"]: batch[1],
                    x["cc24s"]: batch[2]
                })
            train_errors['loss'].append(train_loss_val)
            train_errors['cc_loss'].append(color_constancy_loss_val)
            train_errors['cl_loss'].append(contrastive_loss_val)
        train_error_p = util.percentiles.valuesdict_to_percentilesdict(train_errors)
        print_error_percentile(EXP_NAME, "train", i, train_error_p)

        # Testing. Consistent to eval.py.
        test_errors = OrderedDict([('loss', [])])
        if TEST_PERIOD and i % TEST_PERIOD == 0:
            for _ in tqdm(range(num_test_iterations)):
                batch = sess.run(test_dataloader.get_batch)
                test_loss_val = sess.run(test_loss, feed_dict={
                    x["images"]: batch[0],
                    x["illums"]: batch[1]
                })
                test_errors['loss'].append(test_loss_val)
            test_error_p = util.percentiles.valuesdict_to_percentilesdict(train_errors)
            print('-'*100)
            print(test_error_p['loss'])
            print('-'*100)
            
            # Save
            if trigger.is_best(test_error_p['loss']['mean']):
                ckpt_dir = os.path.join('ckpts', EXP_NAME)
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
            
                ckpt_name = 'MAE_%.3f_%d.ckpt' % (test_error_p['loss']['mean'], i)
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                ckpt_path = saver.save(sess, ckpt_path)
                print('INFO:Save checkpoint to "%s".' % ckpt_path)
    
if __name__ == '__main__':
    with tf.Session() as sess:
        main(sess)