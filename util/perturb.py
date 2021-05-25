import os
import numpy as np
import tensorflow as tf
import cv2
import glob
import csv
import pickle

from easydict import EasyDict
from tqdm import tqdm
from matplotlib import pyplot as plt

slim = tf.contrib.slim

def rand_intensity_gain(image, do_prob, gain_range):
    def _do():
        rand_gain = tf.random_uniform(shape=(), minval=gain_range[0], maxval=gain_range[1], dtype=tf.float32)        
        intensity_gained_image = image * rand_gain
        return intensity_gained_image, rand_gain
    
    rand_prob = tf.random_uniform(shape=(), dtype=tf.float32)
    intensity_gained_image, intensity_gain = tf.cond(tf.less(rand_prob, do_prob), true_fn=_do, false_fn=lambda: (image, 1.0))

    return intensity_gained_image, intensity_gain

def rand_shot_noise(image, do_prob, std_range):
    def _do():
        normalized_image = image / 65535.
        rand_std = tf.random_uniform(shape=(), minval=std_range[0], maxval=std_range[1], dtype=tf.float32)
        noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=rand_std, dtype=tf.float32)
        normalized_noisy_image = normalized_image + tf.sqrt(normalized_image) * noise
        noisy_image = normalized_noisy_image * 65535.
        return noisy_image
    
    rand_prob = tf.random_uniform(shape=(), dtype=tf.float32)
    noisy_image = tf.cond(tf.less(rand_prob, do_prob), true_fn=_do, false_fn=lambda: image)
    
    return noisy_image

def rand_guaussian_noise(image, do_prob, std_range):
    def _do():
        normalized_image = image / 65535.
        rand_std = tf.random_uniform(shape=(), minval=std_range[0], maxval=std_range[1], dtype=tf.float32)
        noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=rand_std, dtype=tf.float32)
        normalized_noisy_image = normalized_image + noise 
        noisy_image = normalized_noisy_image * 65535.
        return noisy_image
        
    rand_prob = tf.random_uniform(shape=(), dtype=tf.float32)
    noisy_image = tf.cond(tf.less(rand_prob, do_prob), true_fn=_do, false_fn=lambda: image)
    
    return noisy_image

def perturb(images):
    images, _ = rand_intensity_gain(images, do_prob=1.0, gain_range=(0.8, 1.2))
    images = rand_shot_noise(images, do_prob=1.0, std_range=(0.02, 0.06))
    images = rand_guaussian_noise(images, do_prob=1.0, std_range=(0.00, 0.04))
    return images