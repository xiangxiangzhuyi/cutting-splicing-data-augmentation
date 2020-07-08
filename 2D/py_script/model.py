# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:09:16 2020

@author: pc
"""

import tensorflow as tf
import unet
import numpy as np
import support_based as spb

class UNET:
    def __init__(self, i_w, i_h, i_ch, o_w, o_h, o_cl, lean_rate, sa_str, train_flag, GPU_str = 'no'):
        # clean graph
        tf.reset_default_graph()
        # data parameters
        self.image_w = i_w
        self.image_h = i_h
        self.image_c = i_ch
        self.mask_w = o_w
        self.mask_h = o_h
        self.class_num = o_cl
        self.learning_rate = lean_rate
        
        # placeholder
        self.input_data = tf.placeholder(tf.float32, [None, self.image_w, self.image_h, self.image_c])
        self.output_mask = tf.placeholder(tf.float32, [None, self.mask_w, self.mask_h, self.class_num])
        self.is_train = tf.placeholder(tf.bool)
        self.lr = tf.placeholder(tf.float32)
        
        # whether GPU
        self.GPU_str = GPU_str
        
        # whether training
        self.tr_flag = train_flag
        self.model_path = spb.mo_path + 'save_results/' + sa_str + '/model/model.ckpt'
        self.sa_str = sa_str
        
    def build_framework(self):
        # Unet
        self.output = unet.Unet(name="UNet", in_data = self.input_data, width = self.image_w,
                                height = self.image_h, channel = self.image_c, cal_num = self.class_num,
                                is_train = self.is_train, reuse = False)
        # loss
        self.loss = unet.uu.generalized_dice_loss(self.output_mask, self.output)
        # optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, name="opt")
            
        self.init = tf.global_variables_initializer()
        
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        
        if self.GPU_str == 'yes':
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        else:
            self.sess = tf.Session()
        
        # whether training the model or load trained model
        if self.tr_flag == 'yes':
            self.sess.run(self.init)
        else:
            self.saver.restore(self.sess, self.model_path)
                
        
    # train
    def train(self, input_data, output_data):
        optim, tr_loss, tr_out = self.sess.run([self.opt, self.loss, self.output], feed_dict={self.input_data: input_data,
                                                self.output_mask: output_data, self.is_train: True, self.lr: self.learning_rate})
        return tr_loss, tr_out
    
    # test
    def test(self, input_data, output_data, b_num):
        los = []
        for ind in range(1000):
            st_num = (ind*b_num)%input_data.shape[0]
            bat_img = input_data[st_num: st_num + b_num, :,:,:]
            bat_mask = output_data[st_num: st_num + b_num, :,:,:]        
            te_loss = self.sess.run([self.loss], feed_dict={self.input_data: bat_img,
                                                self.output_mask: bat_mask, self.is_train: False, self.lr: self.learning_rate})
            los.append(te_loss)
            # check whether ended
            if bat_img.shape[0] < b_num or (ind + 1)*b_num == input_data.shape[0]:
                break
            
        return np.concatenate(los)
    
    # predict
    def pred(self, input_data, output_data, b_num):
        los = []
        sav_ind = 0
        for ind in range(1000):
            st_num = (ind*b_num)%input_data.shape[0]            
            bat_img = input_data[st_num: st_num + b_num, :,:,:]
            bat_mask = output_data[st_num: st_num + b_num, :,:,:]        
            te_loss, te_out = self.sess.run([self.loss, self.output], feed_dict={self.input_data: bat_img,
                                                self.output_mask: bat_mask, self.is_train: False, self.lr: self.learning_rate})
            los.append([te_loss])
            
            # save the result
            for j in range(bat_img.shape[0]):
                spb.save_result(bat_img[j], self.sa_str + '/te_pred/', str(sav_ind) + '_img.npy')
                spb.save_result(bat_mask[j], self.sa_str + '/te_pred/', str(sav_ind) + '_mask.npy')
                spb.save_result(te_out[j], self.sa_str + '/te_pred/', str(sav_ind) + '_pred.npy')
                sav_ind = sav_ind + 1
            
            # check whether ended
            if bat_img.shape[0] < b_num or (ind + 1)*b_num == input_data.shape[0]:
                break
        
        spb.save_result(np.array(los), self.sa_str, 'te_loss.npy')
        
        # calculate DICE
        img_arr, mask_arr, pred_arr = spb.read_pred(self.sa_str)
        pred_mask = spb.pred_to_mask(pred_arr)
        Dice = spb.mean_dice(mask_arr, pred_mask)
        
        return np.concatenate(los), Dice
        
    
    # save the model
    def save(self):
        self.saver.save(self.sess, self.model_path)
        
























