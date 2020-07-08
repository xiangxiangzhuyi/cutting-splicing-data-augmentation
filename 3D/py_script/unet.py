# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:33:43 2020

@author: pc
"""
# this script is used to construct the framework of U-net


# import package
import tensorflow as tf
import unet_util as uu

def Unet(name, in_data, channel, cal_num, is_train, reuse = False):
    
    # construction
    with tf.variable_scope(name, reuse = reuse):
        # down1
        down1 = uu.conv_layer(in_data, [3, 3, 3, channel, 32], 1, 'SAME', is_train)
        down1 = uu.conv_layer(down1, [3, 3, 3, 32, 64], 1, 'SAME', is_train)
        crop1 = down1
        down1 = uu.max_pools3d(down1, 2, 2)
        
        # down2
        down2 = uu.conv_layer(down1, [3, 3, 3, 64, 64], 1, 'SAME', is_train)
        down2 = uu.conv_layer(down2, [3, 3, 3, 64, 128], 1, 'SAME', is_train)
        crop2 = down2
        down2 = uu.max_pools3d(down2, 2, 2)
        
        # down3
        down3 = uu.conv_layer(down2, [3, 3, 3, 128, 128], 1, 'SAME', is_train)
        down3 = uu.conv_layer(down3, [3, 3, 3, 128, 256], 1, 'SAME', is_train)
        crop3 = down3
        down3 = uu.max_pools3d(down3, 2, 2)
        
        # down4
        bottom = uu.conv_layer(down3, [3, 3, 3, 256, 256], 1, 'SAME', is_train)
        bottom = uu.conv_layer(bottom, [3, 3, 3, 256, 512], 1, 'SAME', is_train)

        # up1
        up1 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(bottom)
        up1 = tf.concat([crop3, up1], axis=4)
        up1 = uu.conv_layer(up1, [3, 3, 3, 768, 256], 1, 'SAME', is_train)
        up1 = uu.conv_layer(up1, [3, 3, 3, 256, 256], 1, 'SAME', is_train)
        
        # up2
        up2 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(up1)
        up2 = tf.concat([crop2, up2], axis=4)
        up2 = uu.conv_layer(up2, [3, 3, 3, 384, 128], 1, 'SAME', is_train)
        up2 = uu.conv_layer(up2, [3, 3, 3, 128, 128], 1, 'SAME', is_train)

        # up3
        up3 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(up2)
        up3 = tf.concat([crop1, up3], axis=4)
        up3 = uu.conv_layer(up3, [3, 3, 3, 192, 64], 1, 'SAME', is_train)
        up3 = uu.conv_layer(up3, [3, 3, 3, 64, 64], 1, 'SAME', is_train)
        
        # con 1*1
        out = uu.conv_layer(up3, [1, 1, 1, 64, cal_num], 1, 'SAME', is_train)
        
        return out
        
    
