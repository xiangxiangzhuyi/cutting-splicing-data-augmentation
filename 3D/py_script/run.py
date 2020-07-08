# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:06:59 2020

@author: pc
"""

import model
import support_dataset as spda
import numpy as np
import support_based as spb
import time

def run_model(w_num, h_num, d_num, sam_size,tim_num, tr_bat, te_bat, iter_num, test_once, da_str, GPU_str, aug_str):
    # saving folder
    sa_str = spb.com_mul_str([da_str, sam_size, tim_num, aug_str, w_num, h_num, d_num])
    # construct dataset
    da = spda.dataset(w_num, h_num, d_num, sam_size, tim_num, tr_bat, te_bat, da_str, aug_str)
    
    # save test images and masks
    spb.save_result(da.te_i, sa_str, 'te_img.npy')
    spb.save_result(da.te_m, sa_str, 'te_mask.npy')
        
    # construct UNET
    Unet = model.UNET(da.i_sh, da.o_sh, 0.0001, sa_str, 'yes', GPU_str)
    Unet.build_framework()

    # create lists to save the results
    tr_res = []
    te_res = []
    for i in range(iter_num):
        # train
        tr_img, tr_mask = da.get_tr_bat_img()
        tr_loss, tr_out = Unet.train(tr_img, tr_mask)
        tr_res.append(np.array([i, da.tr_ind, np.mean(tr_loss), tr_img.shape[0]]))
        
        print('train', i, np.mean(tr_loss), time.ctime())
        
        # check 
        flag = spb.ch_glob(tr_res)
        
        # test
        if (i+1)%test_once == 0 or flag:
            te_img, te_mask = da.get_test_bat_img()
            te_loss, te_dice = Unet.pred(te_img, te_mask, tr_bat)
            te_res.append([i, np.mean(te_loss), te_dice])
            
            # print
            print('test', i, np.mean(te_loss), te_dice[-1], time.ctime())
            
            # save result
            spb.save_result(np.stack(tr_res, 0), sa_str, 'tr_res.npy')
            spb.save_list(te_res, sa_str, 'te_res.npy')
            Unet.save()
            
        if flag:
            break
        
    # save result
    spb.save_result(np.stack(tr_res, 0), sa_str, 'tr_res.npy')
    spb.save_list(te_res, sa_str, 'te_res.npy')
    Unet.save()
    
    
    
        
        
