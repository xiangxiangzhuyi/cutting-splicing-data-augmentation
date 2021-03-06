# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:29:34 2020

@author: pc
"""

# This script is used to create sub-sets of the original set

import support_read_img as spri
import support_augdata as spa
import support_based as spb
import numpy as np


# set configuration
# please add or modify command to load your data
i = 0 # the running number
da_na = 'ibsr' # the name of the dataset in string
sam_num = 5 # the sample size of the basic training dataset
im =   # the image numpy array in shape [sample size × 1st dim × 2nd dim × 3rd dim]
ma =  # the segmentation obejct mask in shape [sample size × 1st dim × 2nd dim × 3rd dim]


tr_img, tr_mask, te_img, te_mask = spb.div_set(im, ma, 0.75)
index = np.arange(tr_img.shape[0])
np.random.shuffle(index)

# select images and masks
sel_img = [tr_img[index[x]] for x in range(sam_num)]
sel_mask = [tr_mask[index[x]] for x in range(sam_num)]
sel_img = np.stack(sel_img, 0)
sel_mask = np.stack(sel_mask, 0)

# augmentate data
tr_aug_i, tr_aug_m = spa.aug_img(sel_img, sel_mask, int(10))

# save the data
np_na = spb.mo_path + 'select_data/' + da_na + '_' + str(i) + '_' + str(sam_num) + '_noaug.npz'
np.savez(np_na, tr_i = sel_img, tr_m = sel_mask, te_i = te_img, te_m =te_mask)
np_na = spb.mo_path + 'select_data/' + da_na + '_' + str(i) + '_' + str(sam_num) + '_yesaug.npz'
np.savez(np_na, tr_i = tr_aug_i, tr_m = tr_aug_m, te_i = te_img, te_m =te_mask)
        

        
        
        
        