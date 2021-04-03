# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:07:58 2021

@author: Erik
"""

import os
import numpy as np
import cv2 as cv

def get_tensor(img_path, mask_path, IMG_HEIGHT , IMG_WIDTH):
        
    N_Img = len(os.listdir(img_path))
    
    Images = os.listdir(img_path)
    Mask = os.listdir(mask_path)
    
    #IMG_HEIGHT = 256;
    #IMG_WIDTH = 256;
    
    x_data = np.empty((N_Img, IMG_HEIGHT, IMG_WIDTH, 3), dtype='float32') 
    y_data = np.empty((N_Img, IMG_HEIGHT, IMG_WIDTH, 1), dtype='uint8')
    
    for i in range(0,N_Img):
    
        #A = skimage.io.imread(img_path + Images[i])[:,:,:]
        #B = skimage.io.imread(mask_path + Mask[i])[:,:,:]
        
        A = cv.imread(img_path + Images[i])
        B = cv.imread(mask_path + Mask[i])[:,:,0]
        A = cv.resize(A, (IMG_HEIGHT, IMG_WIDTH))
        B = cv.resize(B, (IMG_HEIGHT, IMG_WIDTH))
        B = B[:,:,np.newaxis]
    
        #if A.shape[0] != IMG_HEIGHT or A.shape[1] != IMG_WIDTH:
        #A = cv.resize(A, dsize=(IMG_HEIGHT, IMG_WIDTH) )
        #B = cv.resize(B, dsize=(IMG_HEIGHT,IMG_WIDTH) )
        
        # A = resize(A, (IMG_HEIGHT , IMG_WIDTH) )
        # B = resize(B, (IMG_HEIGHT , IMG_WIDTH) )
    
        x_data[i] = A
        y_data[i] = B.astype('uint8')
    
    return x_data, y_data

import skimage
from skimage.io import imread

def get_tensor_orig(img_path, mask_path , IMG_HEIGHT , IMG_WIDTH):
    
    N_Img = len(os.listdir(img_path))
  
    Images = os.listdir(img_path)
    Mask = os.listdir(mask_path)

    x_data = np.empty((N_Img, IMG_HEIGHT, IMG_WIDTH, 3), dtype='float32') 
    y_data = np.empty((N_Img, IMG_HEIGHT, IMG_WIDTH, 1), dtype='uint8')

    for i in range(0,N_Img):
        
        A = skimage.io.imread(img_path + Images[i])[:,:,:]
        B = skimage.io.imread(mask_path + Mask[i])[:,:,:]
        
        #if A.shape[0] != IMG_HEIGHT or A.shape[1] != IMG_WIDTH:
        #  A = cv.resize(A, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv.INTER_CUBIC)
        #  B = cv.resize(B, dsize=(IMG_HEIGHT,IMG_WIDTH), interpolation=cv.INTER_CUBIC)
        
        x_data[i] = A
        y_data[i] = B.astype('uint8')

    return x_data, y_data

