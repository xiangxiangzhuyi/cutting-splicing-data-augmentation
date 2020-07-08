# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:00:57 2020

@author: pc
"""
import numpy as np
import cv2
import support_based as spb
import os
import skimage.measure as skm
import nibabel as nib

# get available subjects
def get_avi_li():
    # get the directory
    tr_pa = spb.da_path + 'CHAOS/CHAOS_np_data/Train/MRI/'
    tr_li = os.listdir(tr_pa)
    
    # get the available subject ID
    avi_num = []
    for fi in tr_li:
        fi_num = int(fi.split('_')[0])
        if fi_num not in avi_num:
            avi_num.append(fi_num)
    
    # create all available file names
    all_fi = []
    for fi_num in avi_num:
        all_fi.append([fi_num])
        all_fi[-1].append(str(fi_num) + '_MRI_T1DUAL_InPhase.npy')
        all_fi[-1].append(str(fi_num) + '_MRI_T1DUAL_OutPhase.npy')
        all_fi[-1].append(str(fi_num) + '_MRI_T1DUAL_Ground.npy')
        all_fi[-1].append(str(fi_num) + '_MRI_T2SPIR.npy')
        all_fi[-1].append(str(fi_num) + '_MRI_T2SPIR_Ground.npy')
    
    return all_fi

def read_file(filename, ro_str='no'):
    fi_name = spb.da_path + 'CHAOS/CHAOS_np_data/Train/MRI/' + filename
    img = np.load(fi_name)
    if ro_str == 'yes':
        img = np.rot90(img, k=3, axes=(0, 1))
    return img

# select slices
def res_img(img, mask, r_w, r_h, r_d):
    # check slices are available
    avi_li = []
    for i in range(np.max(mask)):
        ind = np.where(mask == i+1)[2]
        u_ind = np.unique(ind)
        avi_li.append(u_ind)
    a_ind, coun = np.unique(np.concatenate(avi_li), return_counts = True)
    mi_a_ind = np.min(a_ind)
    ma_a_ind = np.max(a_ind) + 1

    # crop the image
    img_c =np.mean(img, axis = 2)
    img_c[img_c < 100] = 0
    img_c[img_c >= 100] = 1
    labeled_img, num = skm.label(img_c, neighbors=8, background=0, return_num=True)
    # get the properities of each region
    reg_li = skm.regionprops(labeled_img)
    proper_li = []
    for reg in reg_li:
        min_row, min_col, max_row, max_col = reg.bbox
        proper_li.append(np.array([min_row, min_col, max_row, max_col]))
        
    r_arr = np.stack(proper_li, 0)
    min_row, min_col = np.min(r_arr[:, 0]), np.min(r_arr[:, 1])
    max_row, max_col = np.max(r_arr[:, 2]), np.max(r_arr[:, 3])
    
    # extracte the image
    cen_row, cen_col = int((min_row + max_row)/2), int((min_col + max_col)/2)
    row_l = int((max_row - min_row)/2) + 0
    col_l = int((max_col - min_col)/2) + 0
    mi_r = max([0, cen_row - row_l])
    ma_r = min([img.shape[0], cen_row + row_l])
    mi_c = max([0, cen_col - col_l])
    ma_c = min([img.shape[1], cen_col + col_l])
    c_img = img[mi_r: ma_r, mi_c: ma_c, mi_a_ind: ma_a_ind]
    c_mask = mask[mi_r: ma_r, mi_c: ma_c, mi_a_ind: ma_a_ind]
    
    # resize the image
    c_img = resize3d(c_img, r_w, r_h, r_d)
    c_mask = resize3d(c_mask, r_w, r_h, r_d)
    
    return c_img, c_mask

# resize 3D image
def resize3d(img, w, h, d):
    #r_img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
    r_img = resize2d(img, w, h)
    r_img = np.transpose(r_img, (2, 1, 0))
    #r_img = cv2.resize(r_img, dsize=(h, d), interpolation=cv2.INTER_NEAREST)
    r_img = resize2d(r_img, d, h)
    
    return np.transpose(r_img, (2, 1, 0))

# resize 2d image
def resize2d(img, w, h):
    re_img = []
    for i in range(img.shape[-1]):
        re_im = cv2.resize(img[:, :, i], dsize=(h, w), interpolation=cv2.INTER_NEAREST)
        re_img.append(re_im)

    return np.stack(re_img, -1)

# normalize to [0, 255]
def nor(img):
    a1 = img - np.min(img)
    a2 = np.max(img) - np.min(img)
    a3 = a1/a2
    a4 = 255*a3
    return a4.astype(np.uint8)

# read brain images
def read_brain():
    pa = spb.da_path + 'IBSR/processed_data/'
    pa_dir = os.listdir(pa)
    im_li = []
    ma_li = []
    for fo in pa_dir:
        i_fi = pa + fo + '/wanat.img'
        im = nib.load(i_fi).get_fdata()
        m_fi = pa + fo + '/wsegm.img'
        ma = nib.load(m_fi).get_fdata()
        im_li.append(im)
        ma_li.append(ma)
    
    # convert mask
    ma_arr = np.stack(ma_li, 0).astype(np.uint8)
    ma_arr[ma_arr == 128] = 1
    ma_arr[ma_arr == 192] = 2
    ma_arr[ma_arr == 253] = 3     
    return np.stack(im_li, 0), ma_arr

# read exist image
def read_ori(da_str):
    if da_str == 't1in':
        data = np.load(spb.da_path + 'CHAOS/3d_np_data/t1_in.npy')
    elif da_str == 't1ou':
        data = np.load(spb.da_path + 'CHAOS/3d_np_data/t1_ou.npy')
    elif da_str == 't2':
        data = np.load(spb.da_path + 'CHAOS/3d_np_data/t2.npy')
    elif da_str == 'ibsr':
        data = np.load(spb.da_path + 'IBSR/np_data/data.npz')
        return nor(data['im']), data['ma'].astype(np.uint8)
    
    return nor(data[:, 0]), data[:, 1].astype(np.uint8)

# read divided dataset
def read_data(da_str, tim, sam, aug_str):
    if aug_str == 'no':
        d = np.load(spb.mo_path + 'select_data/' + da_str + '_' + str(tim) + '_' + str(sam) + '_noaug.npz')
    else:
        d = np.load(spb.mo_path + 'select_data/' + da_str + '_' + str(tim) + '_' + str(sam) + '_yesaug.npz')
    
    # check for covidlesion
    ca_n = np.max(d['tr_m']) + 1

    tr_i = np.expand_dims(d['tr_i'], -1)
    te_i = np.expand_dims(d['te_i'], -1)
    tr_m = spb.conv_one_hot(d['tr_m'], ca_n)
    te_m = spb.conv_one_hot(d['te_m'], ca_n)
    return tr_i, tr_m, te_i, te_m




