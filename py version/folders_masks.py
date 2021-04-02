# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:16:23 2021

@author: Erik
"""

import os
import shutil
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def make_folders():
    
    folders = ['train_imgs', 'valid_imgs', 'test_imgs']
    for folder in folders:
      os.makedirs('/tmp_dataset/' + folder + '/data')
    
    folders = ['train_masks', 'valid_masks', 'test_masks']
    for folder in folders:
      os.makedirs('/tmp_dataset/' + folder + '/data')

def rmv_folders():
    
    folders = ['train_imgs', 'valid_imgs', 'test_imgs']
    for folder in folders:
      shutil.rmtree('/tmp_dataset/' + folder)
    
    folders = ['train_masks', 'valid_masks', 'test_masks']
    for folder in folders:
      shutil.rmtree('/tmp_dataset/' + folder)


def build_mask(array1,array2):
    # Load masks 1
    mask_1 = img_to_array(array1)
    mask_1 = (mask_1/mask_1.max())
    mask_1 = 1.*( mask_1 > 0.5 )
    
    # Load masks 2
    mask_2 = img_to_array(array2)
    mask_2 = mask_2/mask_2.max()
    mask_2 = 2.*( mask_2 > 0.5 )
    
    # Save mask as image uint8
    Sum = mask_1 + mask_2
    Sum[Sum > 2] = 0
    Sum = Sum.astype('uint8')
    
    return Sum

def mask_overlapping(Path_mask_1 , ids_mask_1 , Path_mask_2 , ids_mask_2):
    
    masks_length = list(range(0, len(ids_mask_1)))
    
    Dice_intersection = np.zeros(len(ids_mask_1))
    
    for k in masks_length:
        
        mask_1 = cv.imread(Path_mask_1 + ids_mask_1[k] , cv.IMREAD_GRAYSCALE)
        mask_1 = mask_1/mask_1.max()
        mask_1 = np.array(mask_1).astype(np.bool)
        
        mask_2 = cv.imread(Path_mask_2 + ids_mask_2[k] , cv.IMREAD_GRAYSCALE)
        mask_2 = mask_2/mask_2.max()
        mask_2 = np.array(mask_2).astype(np.bool)
        
        if mask_1.sum() + mask_2.sum() == 0: 
            Dice_intersection = 1
            
        intersection = np.logical_and(mask_1, mask_2)
        Dice_intersection[k] = 2. * intersection.sum() / (mask_1.sum() + mask_2.sum())
        
    plt.figure()
    plt.plot(masks_length , Dice_intersection)
    plt.title('Overlapping degree between two masks')
    plt.ylabel('Dice coef. overlaping')
    plt.xlabel('Mask image dataset')






