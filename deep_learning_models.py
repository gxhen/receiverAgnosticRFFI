
from tensorflow.keras.layers import Input, Lambda, ReLU, Add, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, AveragePooling2D

from tensorflow.keras import layers
import numpy as np
import random

import tensorflow as tf


'''Residual block'''

def resblock(x, kernelsize, filters, first_layer = False):

    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        
        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        
        
        x = Conv2D(filters, 1, padding='same')(x)
        
        out = Add()([x,fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        
        fx = Conv2D(filters, kernelsize, padding='same')(fx)

        out = Add()([x,fx])
        out = ReLU()(out)
        


    return out 




@tf.custom_gradient
def GradientReversalOperator(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradientReversalLayer(Layer):
	def __init__(self,**kwargs):
		super(GradientReversalLayer, self).__init__()
		
	def call(self, inputs):
		return GradientReversalOperator(inputs)


def adv_net(datashape, num_classes, num_rx):
    
    inputs = Input(shape=(52, 126, 1))

    x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)

    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)
    
    x = resblock(x, 3, 64, first_layer = True)
    x = resblock(x, 3, 64)

    x = AveragePooling2D(pool_size=2)(x)
    
    x = Flatten()(x)

    x = Dense(512)(x)
    feature = Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)
    
    y = Dense(128, name = 'tx_clf_input')(feature)

    y = ReLU()(y)
    
    
    tx_out = Dense(num_classes, activation= 'softmax', name = 'tx_classifier')(y)
    
    
    z = GradientReversalLayer()(feature)

    z = Dense(128)(z)
    
    z = ReLU()(z)

    rx_out = Dense(num_rx, activation= 'softmax', name = 'rx_classifier')(z)
    
    
    model = Model(inputs=inputs, outputs=[tx_out,rx_out])
    
    return model  




def non_adv_net(datashape, num_classes):
    
    inputs = Input(shape=(52, 126, 1))
    
    x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)

    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)
    
    x = resblock(x, 3, 64, first_layer = True)
    x = resblock(x, 3, 64)

    x = AveragePooling2D(pool_size=2)(x)
    
    x = Flatten()(x)

    x = Dense(512)(x)
    
    feature = Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)
    
    
    y = Dense(128, name = 'tx_clf_input')(feature)
    

    y = ReLU()(y)
    
    
    tx_out = Dense(num_classes, activation= 'softmax', name = 'tx_classifier')(y)

    
    model = Model(inputs=inputs, outputs=tx_out)
    
    return model  
 


def remove_rx_clf(model):
    encoder = Model(inputs=model.input, outputs= model.get_layer('tx_classifier').output)
    return encoder

