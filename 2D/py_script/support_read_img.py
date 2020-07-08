# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:15:18 2020

@author: pc
"""

# functions to read datasets

import numpy as np
import support_based as spb


# read image by dataset's name
def read_data(da_str, tim, sam, aug_str):
    if aug_str == 'no':
        d = np.load(spb.mo_path + 'select_data/' + da_str + '_' + str(tim) + '_' + str(sam) + '_noaug.npz')
    else:
        d = np.load(spb.mo_path + 'select_data/' + da_str + '_' + str(tim) + '_' + str(sam) + '_yesaug.npz')
    
    # check for covidlesion
    ca_n = np.max(d['tr_m']) + 1
        
    # change shape
    if da_str != 'drive':
        tr_i = np.expand_dims(d['tr_i'], -1)
        te_i = np.expand_dims(d['te_i'], -1)
    else:
        tr_i = d['tr_i']
        te_i = d['te_i']
    tr_m = spb.conv_one_hot(d['tr_m'], ca_n)
    te_m = spb.conv_one_hot(d['te_m'], ca_n)
    return tr_i, tr_m, te_i, te_m


def nor(img):
    a1 = img - np.min(img)
    a2 = np.max(img) - np.min(img)
    a3 = a1/a2
    a4 = 255*a3
    return a4.astype(np.uint8)


