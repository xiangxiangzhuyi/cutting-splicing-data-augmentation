# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:26:15 2020

@author: pc
"""

import os
import support_based as spb
import support_based2 as spb2
import numpy as np

sa_li = os.listdir(spb.mo_path + 'save_results')
conf_li = []
for sa_str in sa_li:
    print(sa_str)
    conf = spb2.convert_str(sa_str)
    all_res, max_dice, last20_dice, mean_dice, tr_tim = spb2.read_result(sa_str)
    conf_li.append(np.concatenate([conf, np.array([max_dice, last20_dice, mean_dice, tr_tim])]))

conf_arr = np.stack(conf_li, 0)
# save the result
filename = spb.mo_path + 'stat_res/sta_res.npy' 
np.save(filename, conf_arr)
