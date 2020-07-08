# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:30:03 2020

@author: pc
"""

import numpy as np
import support_based as spb
import os

# convert string to model configuration
def convert_str(sa_str):
    str_li = sa_str.split('_')
    
    return np.array(str_li[0:-1])

def read_result(sa_str):
    # check whether finished
    fil = spb.mo_path + 'save_results/' + sa_str + '/tr_res.npy'
    if not os.path.isfile(fil):
        return 0,0,0,0,0
    tr_res = spb.read_result(sa_str, 'tr_res.npy')
    te_loss = spb.read_result(sa_str, 'te_loss.npy')
    te_res = spb.read_result(sa_str, 'te_res.npy')
    all_res = [tr_res, te_loss, te_res]
    
    # analysis
    dice_li = np.array([x[2][3] for x in te_res])
    max_dice = np.max(dice_li)
    last20_dice = np.mean(dice_li[-20:])
    mean_dice = np.mean(dice_li)
    tr_tim = tr_res.shape[0]
    
    return all_res, max_dice, last20_dice, mean_dice, tr_tim


def read_for_pvalue(sa_str):
    # check whether finished
    fil = spb.mo_path + 'save_results/' + sa_str + '/tr_res.npy'
    if not os.path.isfile(fil):
        return 0,0,0,0,0
    tr_res = spb.read_result(sa_str, 'tr_res.npy')
    te_loss = spb.read_result(sa_str, 'te_loss.npy')
    te_res = spb.read_result(sa_str, 'te_res.npy')
    all_res = [tr_res, te_loss, te_res]
    
    # analysis
    dice_li = np.array([x[2][3] for x in te_res])
    max_dice = np.max(dice_li)
    last20_dice = np.mean(dice_li[-20:])
    mean_dice = np.mean(dice_li)
    tr_tim = tr_res.shape[0]
    
    # get the best result
    ind = np.argmax(dice_li)
    be_res = te_res[ind][2][1]
    
    return max_dice, last20_dice, mean_dice, tr_tim, be_res


