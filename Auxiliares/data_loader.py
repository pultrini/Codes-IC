import numpy as np
import os
import tensorflow as tf
import random


def rescale_image_range(img, out_min=0.0, out_max=1.0):
    
    '''
        Rescales image intensity range back to [0,1]
    '''

    in_type = str(type(img))
    if 'EagerTensor' in in_type:
        img = img.numpy()
    
    shape = (img.shape[0],img.shape[1])
    img = img.mean(axis=-1)
    in_min = img.min()
    in_max = img.max()

    def new_pixel_value(x, in_min=in_min, in_max=in_max, out_min=out_min, out_max=out_max):
        out_value = (x - in_min) * (out_max - out_min)/(in_max - in_min) + out_min
        return out_value

    resc = list(map(new_pixel_value, img))
    resc = np.reshape(np.array(resc), shape)
    resc = np.stack([resc, resc, resc],-1)

    print('Image rescaled.')

    return resc


def augmenter(img, label=None, do_resc=True):
    '''
        Performs image aumentation per image object.        
    '''
    
    SEED = np.random.randint(0,100)
    
    img_augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode = "horizontal_and_vertical", seed=SEED),
    tf.keras.layers.RandomRotation(0.2, fill_mode='reflect', interpolation='bilinear', seed=SEED),
    tf.keras.layers.RandomContrast(factor=random.random()) 
    ])
    
    aug_img = img_augmenter(img)
    
    # rescaling augmented image to [0,1] range - Need check object type
    #if resc:
    if do_resc:
        in_type = str(type(aug_img))

        if 'tensorflow.python.framework.ops.Tensor' in in_type:
            aug_img = tf.py_function(func=rescale_image_range, inp=[aug_img], Tout=tf.float32)
        else:
            aug_img = rescale_image_range(aug_img)
        
    if label is not None:
        return aug_img, label
    else:
        return aug_img


def square_ratio(image):
    
    '''
        Receives a numpy image (height, width) or (height, width, channels)
        Padd the image to have a square ratio
    '''
    
    shape = image.shape
    print(shape)
    
    # Identifying the dimention to be padded (the shortest one)
    padding_dim = (0 if shape[0] < shape[1] else 1)
    print(padding_dim)
    
    # Identifying the final padded size
    final_dim_size = (shape[1] if shape[0] < shape[1] else shape[0])
    print(final_dim_size)
    
    # calculating the amount of padd before and after the image content in the desired dimensions
    before_pad = int((final_dim_size - shape[padding_dim])/2)
    after_pad = final_dim_size - before_pad - shape[padding_dim]
    print(before_pad, after_pad)
    
    # buiding array of padding:
    # this padding array stores the number of lines to added in each dimensions of the vector
    padding_array = []
    for d in range(len(shape)):
        
        if d == padding_dim:
            # for each dimension we must provide a padding tuple (before_pad, after_pad) 
            padding_array.append((before_pad, after_pad))
        else:
            padding_array.append((0,0))
    
    print(padding_array)
    # perform the padding
    
    return np.pad(image, padding_array, constant_values=0)


def load_and_preprocess_image(image_path, format, input_shape, augment=False, do_resc=True):
    '''
        Loads each image file using TensorFlow API, resizes maintaining the aspect ratio,
        apply square ratio padding, and augments when True.
    '''
    
    img = tf.io.read_file(image_path)
    
    if format == 'bmp':
        img = tf.image.decode_bmp(img, channels=3)
    if format == 'png':
        img = tf.image.decode_png(img, channels=3)
    if format == 'jpg':
        img = tf.image.decode_jpeg(img,channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (input_shape[0], input_shape[1]), preserve_aspect_ratio=True)
    img = square_ratio(img)
    
    if augment:
        img = augmenter(img=img, do_resc=do_resc)
        print('Image augmented.')
    
    img = tf.constant(img, dtype=tf.float32)
    
    return img


def generate_path_dataset(image_tensor_list, classification):
    '''
        Important TIP: always provide paths with fowardslash instead of backslash
        This functions unites a list X inputs with Y label into a daset tensor (X,Y)
    '''
    
    labels = tf.constant(classification)
    dataset = tf.data.Dataset.from_tensor_slices((image_tensor_list, labels))

    return dataset
