# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:40:01 2020

@author: pc
"""

import numpy as np
import support_based as spb
import unet_util as uu
import support_read_img as spri

class dataset:
    def __init__(self, w_num, h_num, d_num, sam_size, tim_num, tr_bat, te_bat, da_str, aug_str):
        # the training set and the test set
        self.tr_i, self.tr_m, self.te_i, self.te_m = spri.read_data(da_str, tim_num, sam_size, aug_str)  
        # input and output img size
        self.i_sh, self.o_sh = self.tr_i.shape[1:], self.tr_m.shape[1:]
        
        # get data parameters
        self.tr_bat = tr_bat
        self.tr_ind = 0
        self.te_bat = te_bat
        self.te_ind = 0
        
        # splicing parameters
        self.w_num = w_num
        self.h_num = h_num
        self.d_num = d_num
        self.sp_num = w_num*h_num*d_num
        
        # training index
        self.tr_index = spb.com_index(self.tr_i.shape[0], self.sp_num)
        
        # test index
        self.te_index = np.arange(self.te_i.shape[0], dtype=np.uint16)
        
    # get the batch index    
    def get_bat_index(self, index_list, bat_num, ind):
        if bat_num >= index_list.shape[0]:
            bat_index = index_list
            ind = 0
        else:
            st_num = (ind*bat_num)%len(index_list)
            bat_index = index_list[st_num: st_num + bat_num]
            if bat_index.shape[0] < bat_num:
                ind = 0
            elif bat_index.shape[0] == bat_num and (ind + 1)*bat_num == len(index_list):
                ind = 0    
            else:
                ind = ind + 1
        return bat_index, ind
    
    # get traning batch images
    def get_tr_bat_img(self):
        bat_index, self.tr_ind = self.get_bat_index(self.tr_index, self.tr_bat, self.tr_ind)
        # images
        bat_img, bat_mask = spb.read_tr_img(self.tr_i, self.tr_m, bat_index, self.w_num, self.h_num, self.d_num)
        
        return bat_img, bat_mask
    
    # get test batch images
    def get_test_bat_img(self):
        bat_index, self.te_ind = self.get_bat_index(self.te_index, self.te_bat, self.te_ind)
        # images
        bat_img = np.stack([self.te_i[x, :,:,:] for x in bat_index], 0)
        # masks
        bat_mask = np.stack([self.te_m[x, :,:,:] for x in bat_index], 0)

        return bat_img, bat_mask
    
    
    