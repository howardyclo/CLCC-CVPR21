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

""" Raw-to-raw color augmentation """

EPS = 1e-7

def mask_paired_cc24s(cc24s_A, cc24s_B):
  # Mask underexposure & overexposure cc24
  # Gehler has average 0.079 gain diff least-square error after tuning threhsolds on fold1 (189)
  # Original error: 0.087
  # Try 3x11?
  underexposure_thr = 1.
  overexposure_thr = 4095.

  cc24s_A_underexposure_mask = tf.less(cc24s_A, underexposure_thr)
  cc24s_A_underexposure_mask = cc24s_A_underexposure_mask[...,0] & cc24s_A_underexposure_mask[...,1] & cc24s_A_underexposure_mask[...,2]
  cc24s_A_underexposure_mask = tf.cast(cc24s_A_underexposure_mask, tf.float32)

  cc24s_A_overexposure_mask = tf.greater_equal(cc24s_A, overexposure_thr)
  cc24s_A_overexposure_mask = cc24s_A_overexposure_mask[...,0] & cc24s_A_overexposure_mask[...,1] & cc24s_A_overexposure_mask[...,2]
  cc24s_A_overexposure_mask = tf.cast(cc24s_A_overexposure_mask, tf.float32)

  cc24s_B_underexposure_mask = tf.less(cc24s_B, underexposure_thr)
  cc24s_B_underexposure_mask = cc24s_B_underexposure_mask[...,0] & cc24s_B_underexposure_mask[...,1] & cc24s_B_underexposure_mask[...,2]
  cc24s_B_underexposure_mask = tf.cast(cc24s_B_underexposure_mask, tf.float32)

  cc24s_B_overexposure_mask = tf.greater_equal(cc24s_B, overexposure_thr)
  cc24s_B_overexposure_mask = cc24s_B_overexposure_mask[...,0] & cc24s_B_overexposure_mask[...,1] & cc24s_B_overexposure_mask[...,2]
  cc24s_B_overexposure_mask = tf.cast(cc24s_B_overexposure_mask, tf.float32)

  cc24s_mask = (1 - cc24s_A_underexposure_mask) * (1 - cc24s_A_overexposure_mask) \
              * (1 - cc24s_B_underexposure_mask) * (1 - cc24s_B_overexposure_mask)

  cc24s_A = cc24s_A * cc24s_mask[...,None]
  cc24s_B = cc24s_B * cc24s_mask[...,None]

  # Normalize color checker (per image, not per pixel), exclude masked pixels
  valid_pixel_count = tf.reduce_sum(cc24s_mask, axis=-1) # (b,)
  norm_cc24s_A = (tf.reduce_sum(cc24s_A, axis=[1,2]) + EPS) / (valid_pixel_count + EPS) # (b,)
  norm_cc24s_B = (tf.reduce_sum(cc24s_B, axis=[1,2]) + EPS) / (valid_pixel_count + EPS) # (b,)
  cc24s_A = cc24s_A / norm_cc24s_A[:,None,None]
  cc24s_B = cc24s_B / norm_cc24s_B[:,None,None]

  return cc24s_A, cc24s_B

def coloraug(images, illums, cc24s, shift_id=1, w=None):
  assert len(cc24s.shape) == 3 # (b, 24, 3)
  # -----------------------------------------------------------------------------
  # Create paired data to interpolate a new data point.
  # -----------------------------------------------------------------------------
  images_A = tf.cast(images, dtype=tf.float32)
  illums_A = tf.cast(illums, dtype=tf.float32)
  cc24s_A = tf.cast(cc24s, dtype=tf.float32)
  masks_A = tf.cast(tf.cast(tf.reduce_sum(cc24s, axis=(1,2)), dtype=tf.bool), dtype=tf.float32)

  images_B = tf.concat([images_A[shift_id:], images_A[0:shift_id]], axis=0)
  illums_B = tf.concat([illums_A[shift_id:], illums_A[0:shift_id]], axis=0)
  cc24s_B = tf.concat([cc24s_A[shift_id:], cc24s_A[0:shift_id]], axis=0)
  masks_B = tf.concat([masks_A[shift_id:], masks_A[0:shift_id]], axis=0)

  # -----------------------------------------------------------------------------
  # Solve color transformation matrices
  # -----------------------------------------------------------------------------
  b = tf.shape(images)[0]
  # C == A when w == 0; # C == B when w == 1.
  # C can be A when C and A has different perturbation augmentation. 
  if not w:
    w = tf.cond(tf.greater_equal(tf.random_uniform(()), 0.5), # Avoid sample [-0.3, 0.3]
                lambda: tf.random_uniform((b,), -5.0, -0.3),
                lambda: tf.random_uniform((b,), 0.3, 5.0)) + EPS # EPS prevents nan when encountering zero cc24.
  else:
    w = tf.ones((b,)) * w

  cc24s_A, cc24s_B = mask_paired_cc24s(cc24s_A, cc24s_B)
  M_AB = tf.linalg.lstsq(cc24s_A, cc24s_B, fast=False) # (b, 3, 3)
  M_BA = tf.linalg.lstsq(cc24s_B, cc24s_A, fast=False) # tf.linalg.inv(M_AB + EPS) causes non-invertible error.

  I = tf.eye(3, batch_shape=[b])
  M_AC = (1 - w[:,None,None]) * I + w[:,None,None] * M_AB
  M_BC = (1 - w[:,None,None]) * I + w[:,None,None] * M_BA

  # -----------------------------------------------------------------------------
  # Color augmentation.
  # 1. AWB: (I_A, l_A), (I_AC, l_AC)
  # 2. Contrastive: Pos(I_A, I_BA); Neg(I_A, I_AC); Neg(I_A, I_BC); w != 1
  # -----------------------------------------------------------------------------
  illums_C = (1 - w[:,None]) * illums_A + w[:,None] * illums_B # (b, 3)
  illums_C = (illums_C + EPS) / (tf.norm(illums_C, axis=-1, keepdims=True) + EPS)
  
  # Prevent residual error on gray color on `image_AC` caused by `M_AC`
  # illums_C_pred = tf.einsum('bx,bxk->bk', illums_A, M_AC) # (b, 3)
  illums_C_pred = tf.matmul(illums_A[:,None,:], M_AC)[:,0,:] # (b, 3)
  illums_C_pred = (illums_C_pred + EPS) / (tf.norm(illums_C_pred, axis=-1, keepdims=True) + EPS)
  
  # This makes gray color in `images_AC to be `illums_C` instead of `illums_C_pred`.
  M_diag_C = tf.matrix_diag((illums_C + EPS) / (illums_C_pred + EPS)) # (b, 3, 3)
  # M_AC = tf.einsum('bij,bjk->bik', M_AC, M_diag_C)
  M_AC = tf.matmul(M_AC, M_diag_C)

  # BE CAREFUL! `IMAGES` IS BGR, `M` IS RGB
  # images_BA = tf.einsum('bijx,bxk->bijk', images_B[...,::-1], M_BA)[...,::-1]
  # images_AC = tf.einsum('bijx,bxk->bijk', images_A[...,::-1], M_AC)[...,::-1]
  images_BA = tf.reshape(tf.matmul(tf.reshape(images_B, (b, -1, 3))[...,::-1], M_BA), (b, tf.shape(images)[1], tf.shape(images)[2], 3))[...,::-1] # Pos
  images_AC = tf.reshape(tf.matmul(tf.reshape(images_A, (b, -1, 3))[...,::-1], M_AC), (b, tf.shape(images)[1], tf.shape(images)[2], 3))[...,::-1] # Hard Neg
  images_BC = tf.reshape(tf.matmul(tf.reshape(images_B, (b, -1, 3))[...,::-1], M_BC), (b, tf.shape(images)[1], tf.shape(images)[2], 3))[...,::-1] # Easy Neg
  
  # Align brightness to anchor images (since applying M will increase or decrease brightness)
  norms_A = tf.reduce_mean(images_A, axis=[1,2,3], keep_dims=True) # (b, 1, 1, 1)
  images_BA = images_BA / tf.reduce_mean(images_BA, axis=[1,2,3], keep_dims=True) * norms_A
  images_AC = images_AC / tf.reduce_mean(images_AC, axis=[1,2,3], keep_dims=True) * norms_A
  images_BC = images_BC / tf.reduce_mean(images_BC, axis=[1,2,3], keep_dims=True) * norms_A
  
  # Make sure in valid range.
  images_BA = tf.clip_by_value(images_BA, 0., 65535.)
  images_AC = tf.clip_by_value(images_AC, 0., 65535.)
  images_BC = tf.clip_by_value(images_BC, 0., 65535.)
  
  # -----------------------------------------------------------------------------
  # Mask loss for new data points that are created by empty/invalid color checker,
  # or M_AB has error compared to residual errors.
  # -----------------------------------------------------------------------------
  loss_masks = masks_A * masks_B # (b,)

  return {
      'images': {
          'A': images_A,
          'B': images_B,
          'BA': images_BA,
          'AC': images_AC,
          'BC': images_BC
      },
      'illums': {
          'A': illums_A,
          'B': illums_B,
          'C': illums_C
      },
      'cc24s': {
        'A': cc24s_A,
        'B': cc24s_B
      },
      'M': {
          'AB': M_AB,
          'BA': M_BA,
          'AC': M_AC,
          'BC': M_BC
      },
      'loss_masks': loss_masks
  }

def construct_contrastive_pairs(images, illums, cc24s, num_negatives):
    # Augment raw-to-raw contrastive pairs
    r2r = coloraug(images, illums, cc24s, shift_id=1)

    # Loss masks for masking invalid color augmented examples.
    contrastive_loss_masks = r2r['loss_masks'] # (b,)

    # Construct anchors
    a_images = r2r['images']['A']

    # Construct easy positives
    easy_p_images = r2r['images']['A']

    # Construct hard positives
    hard_p_images = r2r['images']['BA']

    # Construct N easy & hard negatives
    easy_n_images_list = [r2r['images']['BC']]
    hard_n_images_list = [r2r['images']['AC']]

    for i in range(2, num_negatives+1):
        r2r_ = coloraug(images, illums, cc24s, shift_id=i)
        easy_n_images_list.append(r2r_['images']['BC'])
        hard_n_images_list.append(r2r_['images']['AC'])
    easy_n_images = tf.concat(easy_n_images_list, axis=0) # (N * b, h, w, c)
    hard_n_images = tf.concat(hard_n_images_list, axis=0) # (N * b, h, w, c)
    
    return a_images, easy_p_images, hard_p_images, easy_n_images, hard_n_images, contrastive_loss_masks