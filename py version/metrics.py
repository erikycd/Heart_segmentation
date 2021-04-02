# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:13:46 2021

@author: Erik
"""

import numpy as np


def dice_coeff(y_true, y_pred):
  Dice_class_1 = np.zeros(y_true.shape[0])
  Dice_class_2 = np.zeros(y_true.shape[0])
  for k in range(y_true.shape[0]):

    img1 = np.asarray(y_true[k,:,:,1]).astype(np.bool)
    img2 = np.asarray(y_pred[k,:,:,1]).astype(np.bool)
    if img1.sum() + img2.sum() == 0: return 1
    intersection = np.logical_and(img1, img2)
    Dice_class_1[k] = 2. * intersection.sum() / (img1.sum() + img2.sum())

    img1 = np.asarray(y_true[k,:,:,2]).astype(np.bool)
    img2 = np.asarray(y_pred[k,:,:,2]).astype(np.bool)
    if img1.sum() + img2.sum() == 0: return 1
    intersection = np.logical_and(img1, img2)
    Dice_class_2[k] = 2. * intersection.sum() / (img1.sum() + img2.sum())

  return Dice_class_1, Dice_class_2


def jacard_coeff(y_true, y_pred):

  Jacard_class_1 = np.zeros(y_true.shape[0])
  Jacard_class_2 = np.zeros(y_true.shape[0])

  for k in range(y_true.shape[0]):

    img1 = np.asarray(y_true[k,:,:,1]).astype(np.bool)
    img2 = np.asarray(y_pred[k,:,:,1]).astype(np.bool)
    if img1.sum() + img2.sum() == 0: return 1
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1 , img2)
    Jacard_class_1[k] = intersection.sum() / union.sum()

    img3 = np.asarray(y_true[k,:,:,2]).astype(np.bool)
    img4 = np.asarray(y_pred[k,:,:,2]).astype(np.bool)
    if img3.sum() + img4.sum() == 0: return 1
    intersection = np.logical_and(img3, img4)
    union = np.logical_or(img3 , img4)
    Jacard_class_2[k] = intersection.sum() / union.sum()

  return Jacard_class_1, Jacard_class_2