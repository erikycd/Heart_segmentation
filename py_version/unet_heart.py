# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:52:18 2021

@author: Erik
"""

from tensorflow.keras import layers, models
import tensorflow as tf

def get_unet_heart():
    
    def conv_block(input_tensor, num_filters):
        encoder = layers.SeparableConv2D(num_filters, (3, 3), padding='same', depth_multiplier=1)(input_tensor)
        encoder = layers.BatchNormalization(axis=-1)(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.SpatialDropout2D(0.2)(encoder)
        encoder = layers.SeparableConv2D(num_filters, (3, 3), padding='same', depth_multiplier=1)(encoder)
        encoder = layers.BatchNormalization(axis=-1)(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.SpatialDropout2D(0.2)(encoder)
        return encoder

    def decoder_block(input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor) #, kernel_initializer = keras.initializers.he_uniform())(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization(axis=-1)(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.SeparableConv2D(num_filters, (3, 3), padding='same', depth_multiplier=1)(decoder)
        decoder = layers.BatchNormalization(axis=-1)(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.SpatialDropout2D(0.2)(decoder)
        decoder = layers.SeparableConv2D(num_filters, (3, 3), padding='same', depth_multiplier=1)(decoder)
        decoder = layers.BatchNormalization(axis=-1)(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.SpatialDropout2D(0.2)(decoder)
        return decoder

#%% UNET PRETRAINED

    Resnet = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',  
                                                        input_shape=(256,256,3), pooling=None)
    encoder0 = Resnet.layers[0].output #640 , 64
    encoder1 = Resnet.get_layer('conv1_conv').output #320, 64
    encoder2 = Resnet.get_layer('conv2_block3_1_relu').output #160, 64
    encoder3 = Resnet.get_layer('conv3_block4_1_relu').output #80, 128
    encoder4 = Resnet.get_layer('conv4_block6_1_relu').output #40, 256
    center = Resnet.layers[-1].output #20, 2048
    for i in range( len(Resnet.layers) ):
        Resnet.layers[i].trainable = False
    
    decoder4 = decoder_block(center, encoder4, 256)
    decoder3 = decoder_block(decoder4, encoder3, 128)
    decoder2 = decoder_block(decoder3, encoder2, 64)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 32)
    
    # decoder4 = decoder_block(center, encoder4, 640)
    # decoder3 = decoder_block(decoder4, encoder3, 320)
    # decoder2 = decoder_block(decoder3, encoder2, 160)
    # decoder1 = decoder_block(decoder2, encoder1, 80)
    # decoder0 = decoder_block(decoder1, encoder0, 40)
    
    outputs1 = layers.Conv2D(3, (1, 1), activation = 'softmax')(decoder0)


#%% Compile

    model = models.Model(inputs=[Resnet.input], outputs=[outputs1])
    
    return model



