import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras.models import load_model

import numpy as np

class makemodel:
    def __init__(self):
        self.result = 0

    def base_model(input_data):
        np.random.seed(3)
        
        model = Sequential()
        model.add(Dense(16, activation='sigmoid', input_shape=[input_data.shape[1]], name='layer1')) 
        model.add(Dropout(0.1, name='layer2'))
        model.add(Dense(32, activation='sigmoid', name='layer3'))
        model.add(Dropout(0.1, name='layer4'))
        # model.add(Dense(8, activation='sigmoid', name='layer5'))
        # model.add(Dropout(0.1, name='layer6'))
        model.add(Dense(1, name='final_layer'))
        
        print("Version5")
        return model
    
    def layer1_out(input_data):
        np.random.seed(3)
        
        model = Sequential()                
        model.add(Dense(32, activation='sigmoid', input_shape=[input_data.shape[1]], name='layer1'))
        model.add(Dropout(0.1, name='layer2'))        
        # model.add(Dense(8, activation='sigmoid', name='layer3'))
        # model.add(Dropout(0.1, name='layer4'))
        model.add(Dense(1, name='final_layer'))
        print("Version5")
        return model
    
    def conv2d_base(my_shape): # default shape = (32,32,3)
        np.random.seed(3)
        
        model = Sequential()
        model.add(Conv2D(128, (3, 3), activation='relu', input_shape=my_shape, name='layer1'))
        model.add(MaxPool2D((2, 2), name='layer2'))
        model.add(Conv2D(256, (3, 3), activation='relu', name='layer3'))
        model.add(MaxPool2D((2, 2), name='layer4'))
        model.add(Dropout(0.5, name='layer5'))
        model.add(Conv2D(256, (3, 3), activation='relu', name='layer6'))
        model.add(MaxPool2D((2, 2), name='layer7'))
        model.add(Conv2D(256, (3, 3), activation='relu', name='layer8'))

        model.add(Dropout(0.5, name='layer9'))
        model.add(Conv2D(512, (3, 3), activation='relu', name='layer10'))
        model.add(MaxPool2D((2, 2), name='layer11'))


        model.add(Flatten(name='layer12'))
        model.add(Dense(128, activation='relu', name='layer13'))
        model.add(Dense(64, activation='relu', name='layer14'))
        model.add(Dense(32, activation='relu', name='layer15'))
        model.add(Dense(1, activation='sigmoid', name='layer16'))

        model.summary()
    
        return model
    
    def conv2d_L1_out(my_shape): # default shape = (32,32,3)
        np.random.seed(3)
        
        model = Sequential()        
        model.add(Conv2D(256, (3, 3), activation='relu', input_shape=my_shape, name='layer3'))
        model.add(MaxPool2D((2, 2), name='layer4'))
        model.add(Dropout(0.5, name='layer5'))
        model.add(Conv2D(256, (3, 3), activation='relu', name='layer6'))
        model.add(MaxPool2D((2, 2), name='layer7'))
        model.add(Conv2D(256, (3, 3), activation='relu', name='layer8'))

        model.add(Dropout(0.5, name='layer9'))
        model.add(Conv2D(512, (3, 3), activation='relu', name='layer10')) # 528
        model.add(MaxPool2D((2, 2), name='layer11')) 


        model.add(Flatten(name='layer12'))
        model.add(Dense(128, activation='relu', name='layer13'))
        model.add(Dense(64, activation='relu', name='layer14'))
        model.add(Dense(32, activation='relu', name='layer15'))
        model.add(Dense(1, activation='sigmoid', name='layer16'))

        model.summary()
    
        return model
    
    def mura_conv2d_base(my_shape):
        Inception=keras.applications.InceptionResNetV2(include_top=False,input_shape=my_shape)
        
        input_image=keras.layers.Input(my_shape)
        
        x=Inception (input_image)
        #x=keras.layers.GlobalAveragePooling2D()(x)
        x=keras.layers.Flatten()(x)
        #x=keras.layers.Dense(1024)(x)
        #x=keras.layers.Activation(activation='relu')(x)
        #x= keras.layers.Dropout(0.5)(x)
        x=keras.layers.Dense(256)(x)
        x=keras.layers.Activation(activation='relu')(x)
        x= keras.layers.Dropout(0.5)(x)
        x=keras.layers.Dense(2)(x)
        out=keras.layers.Activation(activation='softmax')(x)
        
        model=Model(inputs=input_image,outputs=out)
        
        return model
    
    
    def model_base(input):
        vgg19 = VGG19(weights = 'imagenet', include_top = False, input_shape =input)
        model = Sequential()
        model.add(Conv2D(3, (3, 3), padding='same', input_shape=input, name='layer'))
        model.add(vgg19)
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        
        return model
    
    def model_server(input):
        vgg19 = VGG19(weights = 'imagenet', include_top = False, input_shape=input)
        model = Sequential()
        model.add(vgg19)
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        
        return model
    
    def m_base(my_shape):
        model = Sequential()
        
        model.add(Conv2D(32, 3, activation='relu', input_shape=my_shape, name='layer1'))
        model.add(MaxPooling2D(name='layer2'))
        model.add(Conv2D(64, 3, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, 3, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation= 'softmax'))
        
     
        return model
    
    def m_server(my_shape):
        model = Sequential()
        
#         model.add(Conv2D(32, 3, activation='relu'))
#         model.add(MaxPooling2D())
        model.add(Conv2D(64, 3, activation='relu', input_shape=my_shape, name='layer3'))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, 3, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation= 'softmax'))
        
        
        return model