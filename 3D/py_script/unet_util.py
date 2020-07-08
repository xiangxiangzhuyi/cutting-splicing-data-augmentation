# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:35:54 2020

@author: pc
"""
# This script incloud some convolutional functions

import tensorflow as tf
import math

# convolutional functions ------------------------------------------------------------------------
def conv_layer(inp, shape, stride, pa_str, is_train):
    w = weight_variable(shape)
    inp = conv3d(inp, w, stride, pa_str)
    inp = tf.layers.batch_normalization(inp, training = is_train)    
    outp = tf.nn.relu(inp)
    return outp
   
def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def conv3d(a, w, s, pa_str):
    return tf.nn.conv3d(a, w, strides=[1, s, s, s, 1], padding= pa_str)
    
def max_pools3d(a, h, s):
    return tf.nn.max_pool3d(a, ksize=[1, h, h, h, 1], strides=[1, s, s, s, 1], padding="SAME")

def generalized_dice_loss(labels, logits):
    smooth = 1e-15
    logits = tf.nn.softmax(logits)
    weights = 1. / (tf.reduce_mean(labels, axis=[0, 1, 2, 3]) ** 2 + smooth)

    numerator = tf.reduce_sum(labels * logits, axis=[0, 1, 2, 3])
    numerator = tf.reduce_sum(weights * numerator)

    denominator = tf.reduce_sum(labels + logits, axis=[0, 1, 2, 3])
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

    up1 = 2*(div(down3))
    up2 = 2*(up1)
    up3 = 2*(up2)
    
    res = up3
    
    dif1 = crop_width(down1, up3)
    dif2 = crop_width(down2, up2)
    dif3 = crop_width(down3, up1)

    return dif1, dif2, dif3, int(res)







