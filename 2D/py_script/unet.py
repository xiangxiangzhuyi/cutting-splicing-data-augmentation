# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:33:43 2020

@author: pc
"""
# this script is used to construct the framework of U-net


# import package
import tensorflow as tf
import unet_util as uu

def Unet(name, in_data, width, height, channel, cal_num, is_train, reuse = False):
    
    # construction
    with tf.variable_scope(name, reuse = reuse):
        # down1
        down1 = uu.conv_layer(in_data, [3, 3, channel, 64], 1, 'SAME', is_train)
        down1 = uu.conv_layer(down1, [3, 3, 64, 64], 1, 'SAME', is_train)
        crop1 = down1
        #crop1 = tf.keras.layers.Cropping2D(cropping=(wid_para[0], hei_para[0]))(down1)
        down1 = uu.max_pools(down1, 2, 2)
        
        # down2
        down2 = uu.conv_layer(down1, [3, 3, 64, 128], 1, 'SAME', is_train)
        down2 = uu.conv_layer(down2, [3, 3, 128, 128], 1, 'SAME', is_train)
        crop2 = down2
        #crop2 = tf.keras.layers.Cropping2D(cropping=(wid_para[1], hei_para[1]))(down2)
        down2 = uu.max_pools(down2, 2, 2)
        
        # down3
        down3 = uu.conv_layer(down2, [3, 3, 128, 256], 1, 'SAME', is_train)
        down3 = uu.conv_layer(down3, [3, 3, 256, 256], 1, 'SAME', is_train)
        crop3 = down3
        #crop3 = tf.keras.layers.Cropping2D(cropping=(wid_para[2], hei_para[2]))(down3)
        down3 = uu.max_pools(down3, 2, 2)
        
        # down4
        down4 = uu.conv_layer(down3, [3, 3, 256, 512], 1, 'SAME', is_train)
        down4 = uu.conv_layer(down4, [3, 3, 512, 512], 1, 'SAME', is_train)
        crop4 = down4
        #crop4 = tf.keras.layers.Cropping2D(cropping=(wid_para[3], hei_para[3]))(down4)
        down4 = uu.max_pools(down4, 2, 2)
        
        # bottom
        bottom = uu.conv_layer(down4, [3, 3, 512, 1024], 1, 'SAME', is_train)
        bottom = uu.conv_layer(bottom, [3, 3, 1024, 1024], 1, 'SAME', is_train)
        
        # up1
        up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(bottom)
        up1 = uu.conv_layer(up1, [2, 2, 1024, 512], 1, 'SAME', is_train)
        up1 = tf.concat([crop4, up1], axis=3)
        up1 = uu.conv_layer(up1, [3, 3, 1024, 512], 1, 'SAME', is_train)
        up1 = uu.conv_layer(up1, [3, 3, 512, 512], 1, 'SAME', is_train)
        
        # up2
        up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(up1)
        up2 = uu.conv_layer(up2, [2, 2, 512, 256], 1, 'SAME', is_train)
        up2 = tf.concat([crop3, up2], axis=3)
        up2 = uu.conv_layer(up2, [3, 3, 512, 256], 1, 'SAME', is_train)
        up2 = uu.conv_layer(up2, [3, 3, 256, 256], 1, 'SAME', is_train)
        
        # up3
        up3 = tf.keras.layers.UpSampling2D(size=(2, 2))(up2)
        up3 = uu.conv_layer(up3, [2, 2, 256, 128], 1, 'SAME', is_train)
        up3 = tf.concat([crop2, up3], axis=3)
        up3 = uu.conv_layer(up3, [3, 3, 256, 128], 1, 'SAME', is_train)
        up3 = uu.conv_layer(up3, [3, 3, 128, 128], 1, 'SAME', is_train)
        
        # up4
        up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(up3)
        up4 = uu.conv_layer(up4, [2, 2, 128, 64], 1, 'SAME', is_train)
        up4 = tf.concat([crop1, up4], axis=3)
        up4 = uu.conv_layer(up4, [3, 3, 128, 64], 1, 'SAME', is_train)
        up4 = uu.conv_layer(up4, [3, 3, 64, 64], 1, 'SAME', is_train)
        
        # con 1*1
        out = uu.conv_layer(up4, [1, 1, 64, cal_num], 1, 'SAME', is_train)
        
        # softmax
        #out = tf.nn.softmax(out)
        
        return out
        
    
