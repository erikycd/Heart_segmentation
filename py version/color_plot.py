# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:17:18 2021

@author: Erik
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# VISUALIZATION OF SAMPLE SEGMENTATION IN COLOR

def mask_color_img(img, mask_1, color_1, mask_2, color_2, alpha):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1]. 

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask_1] = color_1
    out = cv.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)

    out[mask_2] = color_2
    out2 = cv.addWeighted(out, alpha, out, 1 - alpha, 0, out)

    return out2



# VISUALIZATION OF TRAIN TENSOR

def visualize(image_batch, mask_batch=None, pred_batch=None, num_samples=5):
    num_classes = mask_batch.shape[-1] if mask_batch is not None else 0
    fix, ax = plt.subplots(num_classes + 1, num_samples, figsize=(num_samples * 2, (num_classes + 1) * 2))

    for i in range(num_samples):
        ax_image = ax[0, i] if num_classes > 0 else ax[i]
        ax_image.imshow(image_batch[i,:,:,0], cmap='gray')
        ax_image.set_xticks([]) 
        ax_image.set_yticks([])
        
        if mask_batch is not None:
            for j in range(num_classes):
                if pred_batch is None:
                    mask_to_show = mask_batch[i,:,:,j]
                else:
                    mask_to_show = np.zeros(shape=(*mask_batch.shape[1:-1], 3)) 
                    mask_to_show[..., 0] = pred_batch[i,:,:,j] > 0.5
                    mask_to_show[..., 1] = mask_batch[i,:,:,j]
                ax[j + 1, i].imshow(mask_to_show, vmin=0, vmax=1)
                ax[j + 1, i].set_xticks([]) 
                ax[j + 1, i].set_yticks([]) 

    plt.tight_layout()
    plt.show()
