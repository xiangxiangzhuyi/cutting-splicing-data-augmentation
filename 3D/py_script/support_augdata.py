# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:33:26 2020

@author: pc
"""

import numpy as np
import support_read_img as spri
from imgaug import augmenters as iaa

# data augmentation
def img_aug_fun(img):
    img1 = img
    seq = iaa.Sequential([
        #iaa.Fliplr(0.5),
        #iaa.Flipud(0.5),
        iaa.GaussianBlur(sigma=(0, 2.5)),
        iaa.CropAndPad(px=((-2, 2), (-2, 2), (-2, 2), (-2, 2))),
        iaa.PiecewiseAffine(scale=(0.01, 0.05)),
        #iaa.ContrastNormalization((0.75, 1.5)),
        #iaa.CoarseDropout(0.02, size_percent=0.5),
        #iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
        #iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
        iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-5, 5), shear=(-8, 8))
        ])
    img_aug = seq.augment_image(img1)

    return img_aug

# augmentation images
def aug_img(img, mask, aug_num):
    img = spri.nor(img)
    # combine
    img_mask = np.concatenate([img, mask], 3)
    img_li = []
    for i in range(img_mask.shape[0]):
        img_li.append(img_mask[i, :, :, :])
        for j in range(aug_num):
            # aug --> reduce dimension
            img_li.append(img_aug_fun(img_mask[i, :, :, :]))
            
    # convert to array
    img_arr = np.stack(img_li, 0)
    aug_i = img_arr[:, :, :, 0:img.shape[-1]]
    aug_m = img_arr[:, :, :, img.shape[-1]:]

    # diorder
    index = np.arange(aug_i.shape[0])
    np.random.shuffle(index)
    
    return aug_i[index], aug_m[index]




