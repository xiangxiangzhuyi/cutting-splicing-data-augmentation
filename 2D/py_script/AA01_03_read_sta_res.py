# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:31:04 2020

@author: pc
"""

# This script is used to read the stata informaion of all running result

import numpy as np
import support_based as spb

sta = np.load(spb.mo_path + 'stat_res/sta_res.npy')

# sum result of the same configuration
cof_li = []
sta_li = []
for i in range(sta.shape[0]):
    index = -1
    for j in range(len(cof_li)):
        if sta[i, 0] == cof_li[j][0] and sta[i, 1] == cof_li[j][1] and sta[i, 3] == cof_li[j][2]  and sta[i, 4] == cof_li[j][3] and sta[i, 5] == cof_li[j][4]:
            index = j
    if i == 0 or index == -1:
        cof_li.append(sta[i, [0, 1, 3,  4, 5]])
        sta_li.append([sta[i, 6:9]])
    else:
        sta_li[index].append(sta[i, 6:9])

# sum
cof_arr = np.stack(cof_li, 0)
sta_arr = []
for sa in sta_li:
    if len(sa) != 1:
        sta_arr.append(np.mean(np.stack(sa, 0).astype(np.float64), 0))
    else:
        sta_arr.append(sa[0].astype(np.float64))

sta_arr = np.stack(sta_arr, 0)
            
        
        

