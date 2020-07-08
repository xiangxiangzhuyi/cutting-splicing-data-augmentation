# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:35:54 2020

@author: pc
"""
# This script incloud some convolutional functions

import tensorflow as tf
import math
import numpy as np

# convolutional functions ------------------------------------------------------------------------
def conv_layer(inp, shape, stride, pa_str, is_train):
    w = weight_variable(shape)
    inp = conv2d(inp, w, stride, pa_str)
    inp = tf.layers.batch_normalization(inp, training = is_train)
    #bias = bias_variable([shape[-1]])
    #inp = conv2d(inp, w, stride, pa_str) + bias
    
    outp = tf.nn.relu(inp)
    return outp
   
def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def conv2d(a, w, s, pa_str):
    return tf.nn.conv2d(a, w, strides=[1, s, s, 1], padding= pa_str)
    
def max_pools(a, h, s):
    return tf.nn.max_pool(a, ksize=[1, h, h, 1], strides=[1, s, s, 1], padding="SAME")

def avg_pools(a, h1, h2, s):
    return tf.nn.avg_pool(a, ksize=[1, h1, h2, 1], strides=[1, s, s, 1], padding="SAME")

# dice loss function
def dice_loss1(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
    
    return 1 - numerator / denominator

# dice loss
def dice_loss2(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

# loss function 3
def dice_coe_loss(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    '''
    https://tensorlayer.readthedocs.io/en/latest/modules/cost.html#tensorlayer.cost.dice_coe
    '''
    target = tf.nn.softmax(target)
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return 1. -  dice

# other loss function
def other_loss(input_masks, output):
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_masks, logits=output))
    #loss = tf.reduce_mean(tf.squared_difference(input_masks, output))
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(input_masks, output))
    
    return loss

def generalized_dice_loss(labels, logits):
    smooth = 1e-15
    logits = tf.nn.softmax(logits)
    weights = 1. / (tf.reduce_mean(labels, axis=[0, 1, 2]) ** 2 + smooth)

    numerator = tf.reduce_sum(labels * logits, axis=[0, 1, 2])
    numerator = tf.reduce_sum(weights * numerator)

    denominator = tf.reduce_sum(labels + logits, axis=[0, 1, 2])
    denominator = tf.reduce_sum(weights * denominator)

    loss = 1. - 2. * (numerator + smooth) / (denominator + smooth)
    return tf.reduce_mean(loss)

# calculate the cropping width
def crop_width(num1, num2):
    aa = int((num1 - num2)%2)
    bb = int(math.floor((num1 - num2)/2))
    return bb, bb + aa

# calcualte the width and the height ------------------------------------------------------------
def div(num):
    return math.ceil(num/2)

def cal(wid):
    down1 = wid
    down2 = div(down1)
    down3 = div(down2)
    down4 = div(down3)
    up1 = 2*(div(down4))
    up2 = 2*(up1)
    up3 = 2*(up2)
    up4 = 2*(up3)
    res = up4
    
    dif1 = crop_width(down1, up4)
    dif2 = crop_width(down2, up3)
    dif3 = crop_width(down3, up2)
    dif4 = crop_width(down4, up1)

    return dif1, dif2, dif3, dif4, int(res)

# padding images
def pad_img(wid, hei):
    match = [np.array([x, cal(x)[-1]]) for x in range(2000)]
    match = np.stack(match, 0)
    dif = np.abs(match[:, 0] - wid)
    index = np.argmin(dif)
    i_w = match[index, 0]
    m_w= match[index, 1]
    
    dif = np.abs(match[:, 0] - hei)
    index = np.argmin(dif)
    i_h = match[index, 0]
    m_h= match[index, 1]
    
    # calculate padding area
    w_1, w_2 = crop_width(i_w, wid)
    h_1, h_2 = crop_width(i_h, hei)
    
    return i_w, w_1, w_2, m_w, i_h, h_1, h_2, m_h






