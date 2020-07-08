# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:09:20 2020

@author: pc
"""

import run
import argparse

# pass parameters to the script
pa = argparse.ArgumentParser(description='manual to this script')
pa.add_argument('--sam_size', type=int, default = None) # the sameple size of the basic training dataset
pa.add_argument('--w_num', type=int, default = None) # the number of components in the 1st dimension
pa.add_argument('--h_num', type=int, default = None) # the number of components in the 2nd dimension
pa.add_argument('--d_num', type=int, default = None) # the number of compoents in the 3rd dimension
pa.add_argument('--tr_bat', type=int, default = None) # the batch size in training phase
pa.add_argument('--GPU_str', type=str, default = None) # 'yes'/'no', whether use GPU
pa.add_argument('--da_str', type=str, default = None) # the name of the dataset
pa.add_argument('--aug_str', type=str, default = None) # 'yes'/'no', whether use classical augmentation technologies
pa.add_argument('--test_once', type=int, default = None) # how many times do you test after iteraing
pa.add_argument('--tim_num', type=int, default = None) # running number
ar = pa.parse_args()

run.run_model(w_num= ar.w_num, h_num=ar.h_num, d_num=ar.d_num, sam_size=ar.sam_size, 
              tim_num=ar.tim_num, tr_bat=ar.tr_bat, te_bat=1000, GPU_str=ar.GPU_str, 
              iter_num= 20000, test_once=ar.test_once, da_str=ar.da_str, aug_str= ar.aug_str)





