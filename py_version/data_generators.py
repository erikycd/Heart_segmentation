# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:43:31 2021

@author: Erik
"""
from keras.preprocessing.image import ImageDataGenerator


def my_traingenerator(size, batch):
  
    #if SEED == None:
    SEED = 1 #np.random.randint(1,100)
  
    data_gen_image = dict(samplewise_center= True,
                          samplewise_std_normalization = True,
                          # rotation_range= 10, #with no DA = 0
                          # width_shift_range= 0.1, #with no DA = 0
                          # height_shift_range= 0.1, #with no DA = 0
                          # shear_range= 0.05, #with no DA = 0
                          # zoom_range= 0.15 #with no DA = 0
                          )
                        
    data_gen_mask = dict(
                         # rotation_range= 10, #with no DA = 0
                         # width_shift_range= 0.1, #with no DA = 0
                         # height_shift_range= 0.1, #with no DA = 0
                         # shear_range= 0.05, #with no DA = 0
                         # zoom_range= 0.15, #with no DA = 0
                         dtype='float16'
                         )                       

    image_datagen = ImageDataGenerator(**data_gen_image)
    mask_datagen = ImageDataGenerator(**data_gen_mask)

    image_generator = image_datagen.flow_from_directory('/tmp_dataset/train_imgs',
                                                        batch_size = batch,
                                                        target_size = (size , size),
                                                        class_mode = None,
                                                        #color_mode = 'grayscale',
                                                        seed = SEED)                                                    

    mask_generator = mask_datagen.flow_from_directory('/tmp_dataset/train_masks',
                                                      batch_size = batch,
                                                      target_size = (size , size),
                                                      class_mode = None,
                                                      color_mode = 'grayscale',
                                                      seed = SEED)  
  
    while True:
        yield image_generator.next(), mask_generator.next()


def my_validgenerator(size, batch):
  
    #if SEED == None:
    SEED = 1 #np.random.randint(1,100)
  
    data_gen_image = dict(samplewise_center= True,
                          samplewise_std_normalization = True,
                          # rotation_range= 10, #with no DA = 0
                          # width_shift_range= 0.1, #with no DA = 0
                          # height_shift_range= 0.1, #with no DA = 0
                          # shear_range= 0.05, #with no DA = 0
                          # zoom_range= 0.15 #with no DA = 0
                          ) 
                        
    data_gen_mask= dict(
                        # rotation_range= 10, #with no DA = 0
                        # width_shift_range= 0.1, #with no DA = 0
                        # height_shift_range= 0.1, #with no DA = 0
                        # shear_range= 0.05, #with no DA = 0
                        # zoom_range= 0.15, #with no DA = 0
                        dtype='float16'
                        )                       

    image_datagen = ImageDataGenerator(**data_gen_image)
    mask_datagen = ImageDataGenerator(**data_gen_mask)

    image_generator = image_datagen.flow_from_directory('/tmp_dataset/valid_imgs',
                                                        batch_size = batch,
                                                        class_mode = None,
                                                        target_size = (size , size),
                                                        #color_mode = 'grayscale',
                                                        seed = SEED)                                                    

    mask_generator = mask_datagen.flow_from_directory('/tmp_dataset/valid_masks',
                                                      batch_size = batch,
                                                      target_size = (size, size),
                                                      class_mode = None,
                                                      color_mode = 'grayscale',
                                                      seed = SEED)  
  
    while True:
        yield image_generator.next(), mask_generator.next()
        
def my_testgenerator(size):
    
    data_gen_image = dict(samplewise_center= True,
                        samplewise_std_normalization = True,
                        # rotation_range= 10, #with no DA = 0
                        # width_shift_range= 0.1, #with no DA = 0
                        # height_shift_range= 0.1, #with no DA = 0
                        # shear_range= 0.05, #with no DA = 0
                        # zoom_range= 0.15 #with no DA = 0
                        )
    data_gen_mask= dict(
                      # rotation_range= 10, #with no DA = 0
                      # width_shift_range= 0.1, #with no DA = 0
                      # height_shift_range= 0.1, #with no DA = 0
                      # shear_range= 0.05, #with no DA = 0
                      # zoom_range= 0.15, #with no DA = 0
                      dtype='uint8'
                      )
    
    image_datagen = ImageDataGenerator(**data_gen_image)
    mask_datagen = ImageDataGenerator(**data_gen_mask)
    
    image_generator = image_datagen.flow_from_directory('/tmp_dataset/test_imgs/',
                                                      class_mode = None,
                                                      shuffle = False,
                                                      target_size = (size , size))
    
    mask_generator = mask_datagen.flow_from_directory('/tmp_dataset/test_masks/',
                                                      class_mode = None,
                                                      color_mode = 'grayscale',
                                                      shuffle = False,
                                                      target_size = (size , size))  
    Test_names = image_generator.filenames  
    
    while True:
      yield Test_names, image_generator.next(), mask_generator.next()
      
