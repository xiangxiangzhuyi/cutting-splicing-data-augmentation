# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:26:19 2020

@author: pc
"""

import run

# pass parameters to the script

run.run_model(w_num=1, h_num= 1, d_num=1, sam_size=5, tim_num=0, 
              tr_bat=1, te_bat=1000, GPU_str='no', iter_num=100000, 
              test_once=3, da_str='ibsr', aug_str='no')
