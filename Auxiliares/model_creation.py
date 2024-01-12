import numpy as np
import os
import tensorflow as tf
import random


def create_model(input_size=(224, 224, 3), choice_model=None, drop_prob=0.2, num_classes=1, pred_acticvation='sigmoid'):
    
    '''
        Creates an instance model according to the baseline architecture chosen.
    '''

    inputs = tf.keras.layers.Input(shape = input_size)
    
    base_model = None
    base_model_input = None
        
    if choice_model == 'inception':
        
        IMG_SHAPE = (299,299,3)
        base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False, weights='imagenet')
        
        # preprocessing to inception input format
        base_model_input = tf.keras.layers.experimental.preprocessing.Resizing( 299, 299, interpolation='bilinear')(inputs)
        base_model_input = tf.keras.applications.inception_v3.preprocess_input(base_model_input)

    
    elif choice_model == 'efficientnetv2':
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(input_shape=input_size,
                                               include_top=False, weights='imagenet')
        # resizing and preprocessing image to MobileNet input dims
        #base_model_input = tf.keras.layers.experimental.preprocessing.Resizing( 224, 224, interpolation='bilinear')(inputs)
        base_model_input = inputs
        base_model_input = tf.keras.applications.efficientnet.preprocess_input(base_model_input)


    elif choice_model == 'inception_rn':
        
        IMG_SHAPE = (299,299,3)
        base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False, weights='imagenet')
        # preprocessing to inception input format
        base_model_input = tf.keras.layers.experimental.preprocessing.Resizing( 299, 299, interpolation='bilinear')(inputs)
        base_model_input = tf.keras.applications.inception_resnet_v2.preprocess_input(base_model_input)
      

    elif choice_model == 'mobile_net':
        
        IMG_SHAPE = input_size
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False, weights='imagenet')
        
        # resizing and preprocessing image to ResNet input dims
        base_model_input = tf.keras.applications.mobilenet_v2.preprocess_input(base_model_input)


    elif choice_model == 'resnet101':
        
        IMG_SHAPE = input_size
        base_model = tf.keras.applications.resnet.ResNet101(input_shape=IMG_SHAPE,
                                                   include_top=False, weights='imagenet')
        
        # resizing and preprocessing image to ResNet input dims
        base_model_input = inputs
        base_model_input = tf.keras.applications.resnet.preprocess_input(base_model_input)
        
    elif choice_model == 'VGG16':
        IMG_SHAPE = (224,224,3)
        
        base_model = tf.keras.applications.vgg16.VGG16(input_shape = IMG_SHAPE,
                                                       include_top = False, weights='imagenet')
        base_model_input = inputs
        base_model_input = tf.keras.applications.vgg16.preprocess_input(base_model_input)



    # Passing through base_model
    base_model_output = base_model(base_model_input)
    
    # Shortening to prediction output
    global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model_output)
    drop = tf.keras.layers.Dropout(drop_prob)(global_average_pooling)
    
    
    dense = tf.keras.layers.Dense(680, activation='relu')(drop)
    drop = tf.keras.layers.Dropout(drop_prob)(dense)
    
    dense = tf.keras.layers.Dense(128, activation='relu')(drop)
    drop = tf.keras.layers.Dropout(drop_prob)(dense)
    
    prediction = tf.keras.layers.Dense(
        num_classes, activation=pred_acticvation)(drop)
    
    outputs = prediction
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=choice_model)   
    
    return model


def start_model(input_size=(224, 224, 3), choice_model=None, choice_opt=None, lr=0.0001, num_classes=1, drop_prob=0.2, pred_acticvation='sigmoid', loss='binary_crossentropy', metrics=['accuracy']):
    
    '''
        Starts a model defining optimizer, learning rate, loss function, and metrics.
    '''

    model = create_model(input_size=input_size, choice_model=choice_model, drop_prob=drop_prob, num_classes=num_classes, pred_acticvation=pred_acticvation)
    
    opt = None
    
    if choice_opt == 'Adam' or choice_opt == None :
        # setting optmizer and compiling
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    
    elif choice_opt == 'Adagrad':
        # setting optmizer and compiling
        opt = tf.keras.optimizers.Adagrad(learning_rate=lr)

    elif choice_opt == 'SGD':
        # setting optmizer and compiling
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    
    elif choice_opt == 'RMSprop':
        # setting optmizer and compiling
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        
    elif choice_opt == 'Adamax':
        # setting optmizer and compiling
        opt = tf.keras.optimizers.Adamax(learning_rate=lr)
    
    elif choice_opt == 'Nadam':
        # setting optmizer and compiling
        opt = tf.keras.optimizers.Nadam(learning_rate=lr)

    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    return model


def set_check_poin_saver(exp_path, exp_name):

    checkpoint_path = os.path.join(exp_path, exp_name + '/cp-{epoch:04d}.ckpt')
    #os.makedirs(checkpoint_path, exist_ok=True)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor = 'val_accuracy',
                                                     save_best_only = True,
                                                     save_weights_only=True,
                                                     mode='max',
                                                     verbose=1)
    
    return cp_callback

def set_check_point_saver_regression(exp_path, exp_name):

    checkpoint_path = os.path.join(exp_path, exp_name + '/cp-{epoch:04d}.ckpt')
    #os.makedirs(checkpoint_path, exist_ok=True)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor = 'val_mae',
                                                     save_best_only = True,
                                                     save_weights_only=True,
                                                     mode='min',
                                                     verbose=1)
    
    return cp_callback