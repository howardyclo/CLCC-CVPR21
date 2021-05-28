""" Refactor from FC4 augmentation on numpy data. 
    Recommended to use tensorflow native operations.
"""

import cv2
import math
import random
import numpy as np
import util.rotate
from config import *

def rand_crop_square(image):
    scale = math.exp(random.random() * math.log(
      AUGMENTATION_SCALE[1] / AUGMENTATION_SCALE[0])) * AUGMENTATION_SCALE[0]
    s = int(round(min(image.shape[:2]) * scale))
    s = min(max(s, 10), min(image.shape[:2]))
    start_x = random.randrange(0, image.shape[0] - s + 1)
    start_y = random.randrange(0, image.shape[1] - s + 1)
    return image[start_x:start_x + s, start_y:start_y + s]

def rand_rotate_and_crop(image):
    angle = (random.random() - 0.5) * AUGMENTATION_ANGLE
    return util.rotate.rotate_and_crop(image, angle)
    
def rand_flip(image):
    if random.randint(0, 1):
        image = image[:, ::-1] # Left-right
    if random.randint(0, 1):
        image = image[::-1, :] # Top-down
    return image
    
def mask_overexposure(image, value=65535.):
    binary_mask = (image >= value)
    binary_mask_channelwise = binary_mask[...,0] | binary_mask[...,1] | binary_mask[...,2]
    binary_mask_channelwise = binary_mask_channelwise[...,None]
    repeat_mask = np.repeat(binary_mask_channelwise, 3, axis=-1)
    image[repeat_mask] = 0.
    return image

def rand_intensity_gain_and_mask_overexposure(image):
    if random.random() < 0.5:
        image = mask_overexposure(image, 65535.)
        image = image * random.uniform(AUGMENTATION_GAIN[0], AUGMENTATION_GAIN[1])
        image = mask_overexposure(image, 65535.)
    return image
    
def rand_coloraug(image, illum, cc24):
    """ Transformation-equivariant augmentation.
        I.e., (x, y) => (aug(x), aug(y)).
        * image: BGR [0, 65535] (h, w, 3)
        * illum: RGB (3,)
        * cc24: RGB [0, 4095] (24, 3)
    """
    # Perturbation coefficients on RGBs.
    color_aug = np.zeros(shape=(3,))
    for i in range(3):
        color_aug[i] = 1 + random.random() * AUGMENTATION_COLOR - 0.5 * AUGMENTATION_COLOR
    
    # Perturb RGB
    new_image = image * np.array([color_aug[0], color_aug[1], color_aug[2]])[None, None, ::-1]
    new_image = np.clip(new_image, 0, 65535)
    new_illum = illum * np.array([color_aug[0], color_aug[1], color_aug[2]])
    new_illum = new_illum / np.linalg.norm(new_illum, axis=-1, keepdims=True)
    new_cc24 = cc24 * np.array([color_aug[0], color_aug[1], color_aug[2]])[None, :]
    new_cc24 = np.clip(new_cc24, 0, 4095)
    
    return new_image, new_illum, new_cc24
    
def augment(image, illum, cc24):
    """ * image: BGR [0, 65535] (h, w, 3)
        * illum: RGB (3,)
        * cc24: RGB [0, 4095] (24, 3)
    """
    image = rand_crop_square(image)
    image = rand_rotate_and_crop(image)
    image = cv2.resize(image, (TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE))
    image = rand_flip(image)
    image = rand_intensity_gain_and_mask_overexposure(image)
    image, illum, cc24 = rand_coloraug(image, illum, cc24)
    return image, illum, cc24