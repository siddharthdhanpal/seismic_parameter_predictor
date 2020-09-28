#!/usr/bin/env python
# coding: utf-8
import errno
import tensorflow as tf
import os
import os.path
import numpy as np
import skimage.measure

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Activation,Dropout,MaxPooling1D,AveragePooling1D,BatchNormalization
from tensorflow.keras.layers import Input,Conv1D,SeparableConv1D,LSTM,GRU,Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

import time
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_yaml


# Using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# creating directory and path 
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

num_train      = 90000
num_test       = 5000
num_validation = 5000
examples       = num_train + num_validation + num_test

num_classes    = 50
num_epochs     = 25
num_batchsize  = 128
loss           = 'sparse_categorical_crossentropy'
pos_enc        = False #False/True

path = './../models/test_git_code_parameter_classifier'


mkdir_p(path)
print('path created')

#loading X data
start_time=time.time()
print(start_time)
X = np.load('./../data_regression/data_100k.npy')
end_time=time.time()
print(X.shape)
print(end_time)
print('Time taken for loading data= %f s'%(end_time-start_time))


Y = np.loadtxt('./../data_regression/data_y_100k.npy')



print(X.shape,Y.shape)


print('dataset preparation completed')
# 5 parameters among these Y. (average core rotation, average env rot, Delta nu, Delta pi, coupling factor q)
aer, acr, Dnu, Dp, epsilon_g ,q = Y[:,0], Y[:,1], Y[:,2], Y[:,7], Y[:,8], Y[:,9] 

inc = Y[:,-1]

epsilon_p = Y[:,3] 

snr = Y[:,10]

sample_weights = np.ones(Dnu.shape)

#Scale parameters in between [0,1]
def scale_parameter(a):
    a = (a-a.min())/(a.max()-a.min())
    return a

aer = scale_parameter(aer)
acr = scale_parameter(acr)
Dnu = scale_parameter(Dnu)
Dp  = scale_parameter(Dp)
q   = scale_parameter(q)
epsilon_g = scale_parameter(epsilon_g)
epsilon_p = scale_parameter(epsilon_p)
inc = scale_parameter(inc)


parameter = Dnu
parameter = parameter*num_classes
parameter = np.floor(parameter)
parameter = parameter.astype('int')
parameter = np.where(parameter<num_classes,parameter,num_classes-1)

def positional_enc(X):

    X=np.squeeze(X) 
    X = 2.*X 
    X = X-1.
    pos_enc_sin = np.sin(np.arange(X.shape[1]))
    pos_enc_cos = np.cos(np.arange(X.shape[1]))


    X_pos    = np.empty((X.shape[0],X.shape[1],2),dtype=np.float32)
    X_pos[:,:,0] = X + pos_enc_sin
    X_pos[:,:,1] = X + pos_enc_sin

    return X_pos

X = X.reshape(X.shape[0],X.shape[1],1)

if pos_enc ==True:
    X = positional_enc(X)

#Splitting each label/dataset accordingly

X_train = X[:num_train]
X_val   = X[num_train:num_train+num_validation]
X_test  = X[num_train+num_validation:]


parameter_train = parameter[:num_train]
parameter_val   = parameter[num_train:num_train+num_validation]
parameter_test  = parameter[num_train+num_validation:]

sample_weights_train = sample_weights[:num_train]
sample_weights_val   = sample_weights[num_train:num_train+num_validation]

#This list depends on the number of parameters
labels_train = parameter_train
labels_val   = parameter_val


'''Metrics'''
#Pearson correlation coefficient: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
def pearson_correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    return r

#Coefficient of determination (r_square): https://en.wikipedia.org/wiki/Coefficient_of_determination
def r2(y_true, y_pred):
    x = y_true
    y = y_true-y_pred
    mx = K.mean(x)
    xm, ym = x-mx, y
    r_den = K.sum(tf.multiply(xm,xm))
    r_num = K.sum(tf.multiply(ym,ym))
    r = 1.-(r_num / r_den)

    return r

#Explained variance : https://en.wikipedia.org/wiki/Explained_variation
def explained_variance(y_true, y_pred):
    x = y_true
    y = y_true-y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_den = K.sum(tf.multiply(xm,xm))
    r_num = K.sum(tf.multiply(ym,ym))
    r = 1.-(r_num / r_den)

    return r


'''Different Models'''
#LSTM model
def lstm_model(X,pos_enc=False):
    kernel_size = 5
    strides = 1
    pool_size = 3
    filters = 4
    dilation_rate=1

    visible = Input(shape=(X.shape[1],1))
    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible
    for j in range(0,2):    #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            #x =  Conv1D(filters = filters,kernel_size = kernel_size, dilation_rate=dilation_rate, strides=1, padding='same', activation='relu')(x)
            x = LSTM(filters, return_sequences=True)(x)
            #x =  BatchNormalization()(x)
            filters = 2*filters
        #x = MaxPooling1D(pool_size = pool_size, strides=1, padding='same')(x)
        #x = AveragePooling1D(pool_size=pool_size, padding='same')(x)
        if j>=1:
            x = Dropout(0.25)(x)
    #x = Dense(1000, activation='relu')(x)
    #x = Dropout(0.25)(x)
    flat = Flatten(name='flatten')(x)

    output1 = Dense(1, activation='linear', name='Dp')(flat)#kernel_regularizer=regularizers.l2(0.001)
    output2 = Dense(1, activation='linear', name='Dnu')(flat)
    output3 = Dense(1, activation='linear', name='q')(flat)
    output4 = Dense(1, activation='linear', name='aer')(flat)
    output5 = Dense(1, activation='linear', name='acr')(flat)
    output6 = Dense(1, activation='linear', name='epsilon_p')(flat)
    #output7 = Dense(1, activation='linear', name='epsilon_g')(flat)

    output = output6
    #output  = [output1,output2,output3,output4,output5,output6] #Add additional outputs/parameters according to requirements

    model = Model(inputs=visible, outputs=output)
    #plot_model(model, to_file='%s/model.png'%path,show_shapes=True)
    #print(model.summary())
    return model


#CNN Model

def cnn_lstm_model(X,num_classes,pos_enc=False):
    '''
    Defining CNN-LSTM-Dense model with kernel size, pool size, strides,
    filters and dilation rate given in this function.

    Input: spectrum of shape (None,length of spectrum,1) and number of bins
    in parameter space

    Returns a CNN-LSTM-Dense model.
    
    '''
    kernel_size = 5
    strides = 1
    pool_size = 3
    filters = 16
    dilation_rate=1

    visible = Input(shape=(X.shape[1],1))

    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible


    #Conv1D model

    for j in range(0,6):    #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  Conv1D(filters = filters,kernel_size = kernel_size, dilation_rate=dilation_rate, strides=1, padding='same', activation='relu')(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        x = MaxPooling1D(pool_size = pool_size, strides=1, padding='same')(x)
        x = AveragePooling1D(pool_size=pool_size, padding='same')(x)
        if j>=4:
            x = Dropout(0.25)(x)
    
    #LSTM model
    x = LSTM(256,return_sequences=True)(x)
    x = Dropout(0.25)(x)
   
    x = LSTM(512,return_sequences=True)(x)
    x = Dropout(0.25)(x)
    flat = Flatten(name='flatten')(x)

    #Dense layer
    flat = Dense(200, activation='tanh')(flat)

    flat = Dropout(0.25)(flat)

    output1 = Dense(num_classes, activation='softmax', name='parameter')(flat)
    output = output1

    model = Model(inputs=visible, outputs=output)
    return model

def cnn_lstm_attention_model(X,num_classes,pos_enc=False):
    '''
    Defining CNN-Attention-LSTM-Dense model with kernel size, pool size, strides,
    filters and dilation rate given in this function.

    Input: spectrum of shape (None,length of spectrum,1) and number of bins
    in parameter space

    Returns a CNN-Attention-LSTM-Dense model.
    
    '''

    kernel_size = 5
    strides = 1
    pool_size = 3
    filters = 16
    dilation_rate=1

    visible = Input(shape=(X.shape[1],1))

    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible

    #Conv1D network

    for j in range(0,6):    #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  Conv1D(filters = filters,kernel_size = kernel_size, dilation_rate=dilation_rate, strides=1, padding='same', activation='relu')(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        x = MaxPooling1D(pool_size = pool_size, strides=1, padding='same')(x)
        x = AveragePooling1D(pool_size=pool_size, padding='same')(x)
        if j>=4:
            x = Dropout(0.25)(x)

    #Attention layer

    x  = tf.keras.layers.AdditiveAttention(causal=False)([x,x])

    #LSTM layers
    x = LSTM(256, dropout=0.2 ,recurrent_dropout=0.2 ,return_sequences=True)(x)
   
    x = LSTM(512,dropout=0.2 ,recurrent_dropout=0.2 ,return_sequences=True)(x)



    flat = Flatten(name='flatten')(x)

    #Dense layer

    flat = Dense(500, activation='tanh')(flat)
    flat = Dropout(0.25)(flat)

    output1 = Dense(num_classes, activation='softmax', name='parameter')(flat)

    output = output1

    model = Model(inputs=visible, outputs=output)
    return model




def separable_cnn_model(X,pos_enc=False):
    kernel_size = 5
    strides = 1
    pool_size = 5
    filters = 16

    visible = Input(shape=(X.shape[1],1))
    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible
    for j in range(0,6):  #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  SeparableConv1D(filters = filters,kernel_size = kernel_size, strides=1, padding='same', activation='tanh')(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        x = MaxPooling1D(pool_size = pool_size, strides=1, padding='same')(x)
        x = AveragePooling1D(pool_size=pool_size, padding='same')(x)
        if j>=4:
            x = Dropout(0.25)(x)

    x = Dropout(0.25)(x)
    flat = Flatten(name='flatten')(x)

    output1 = Dense(1, activation='linear', name='Dp')(flat)#kernel_regularizer=regularizers.l2(0.001)
    output2 = Dense(1, activation='linear', name='Dnu')(flat)
    output3 = Dense(1, activation='linear', name='q')(flat)
    output4 = Dense(1, activation='linear', name='aer')(flat)
    output5 = Dense(1, activation='linear', name='acr')(flat)
    #output6 = Dense(1, activation='linear', name='epsilon_p')(flat)
    #output7 = Dense(1, activation='linear', name='epsilon_g')(flat)

    #output = output1
    output  = [output1,output2,output3,output4,output5]#,output6]#,output7]

    model = Model(inputs=visible, outputs=output)
    #plot_model(model, to_file='%s/model.png'%path,show_shapes=True)
    #print(model.summary())
    return model

def pure_cnn_model(X,pos_enc=False):
    kernel_size = 20
    strides = 1
    pool_size = 5
    filters = 16
    stride_pool = 3
    dilation_rate= 1

    visible = Input(shape=(X.shape[1],1))
    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible
    for j in range(0,7): #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  Conv1D(filters = filters,kernel_size = kernel_size, dilation_rate=dilation_rate, strides=1, padding='same', activation='relu')(x)
            x =  BatchNormalization()(x)
            x =  Conv1D(filters = filters,kernel_size = kernel_size, strides=stride_pool, padding='same', activation='relu')(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        if j>=6:
            x = Dropout(0.25)(x)

    #x = Dropout(0.25)(x)
    flat = Flatten(name='flatten')(x)

    output1 = Dense(1, activation='linear', name='Dp')(flat)#kernel_regularizer=regularizers.l2(0.001)
    output2 = Dense(1, activation='linear', name='Dnu')(flat)
    output3 = Dense(1, activation='linear', name='q')(flat)
    output4 = Dense(1, activation='linear', name='aer')(flat)
    output5 = Dense(1, activation='linear', name='acr')(flat)
    #output6 = Dense(1, activation='linear', name='epsilon_p')(flat)
    #output7 = Dense(1, activation='linear', name='epsilon_g')(flat)

    #output = output1
    output  = [output1,output2,output3,output4,output5]#,output6]#,output7]

    model = Model(inputs=visible, outputs=output)
    #plot_model(model, to_file='%s/model.png'%path,show_shapes=True)
    #print(model.summary())
    return model

def pure_separable_cnn_model(X,pos_enc=False):
    kernel_size = 5
    strides = 1
    pool_size = 5
    filters = 16
    stride_pool = 5

    visible = Input(shape=(X.shape[1],1))
    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible
    for j in range(0,6): #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  SeparableConv1D(filters = filters,kernel_size = kernel_size, strides=1, padding='same', activation='tanh')(x)
            x =  BatchNormalization()(x)
            x =  SeparableConv1D(filters = filters,kernel_size = kernel_size, strides=stride_pool, padding='same', activation='tanh')(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        if j>=4:
            x = Dropout(0.25)(x)

    #x = Dropout(0.25)(x)
    flat = Flatten(name='flatten')(x)

    output1 = Dense(1, activation='linear', name='Dp')(flat)#kernel_regularizer=regularizers.l2(0.001)
    output2 = Dense(1, activation='linear', name='Dnu')(flat)
    output3 = Dense(1, activation='linear', name='q')(flat)
    output4 = Dense(1, activation='linear', name='aer')(flat)
    output5 = Dense(1, activation='linear', name='acr')(flat)
    output6 = Dense(1, activation='linear', name='epsilon_p')(flat)
    #output7 = Dense(1, activation='linear', name='epsilon_g')(flat)

    #output = output1
    output  = [output1,output2,output3,output4,output5,output6]#,output7]

    model = Model(inputs=visible, outputs=output)
    #plot_model(model, to_file='%s/model.png'%path,show_shapes=True)
    #print(model.summary())
    return model



model = cnn_lstm_model(X,num_classes,pos_enc)

# Script to load a model and load weights
'''
yaml_file = open('%s/model.yaml'%path, 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("%s/model.h5"%path)
print("Loaded model from disk")
'''
print(model.summary())

def scheduler(epoch):
    if epoch <= 5: 
        return 0.0003
    elif epoch<=8:
        return 0.00018
    elif epoch<=15:
        return 0.0001
    elif epoch<=20:
        return 0.00005
    else:
        return 0.00001 


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


checkpoint = ModelCheckpoint(filepath= path+'/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',period=7)

callbacks = [callback,checkpoint]
model_yaml = model.to_yaml()
with open("%s/model.yaml"%path, "w") as yaml_file:
    yaml_file.write(model_yaml)

#Compiling model
model.compile(loss=loss,optimizer=tf.keras.optimizers.Adam(),metrics=['sparse_categorical_accuracy'])

start_time = time.time()
history=model.fit(X_train, labels_train, epochs=num_epochs,validation_data=(X_val, labels_val), batch_size=num_batchsize,initial_epoch=0, verbose=1,sample_weight=sample_weights_train,callbacks=callbacks)
end_time = time.time()
print('Time taken for training= %f s'%(end_time-start_time))

#Writing predictions
predictions = model.predict(X)
predictions = np.array(predictions)
predictions = np.squeeze(predictions)
np.save('%s/predicted_probabilities.npy'%path,predictions)
np.savetxt('%s/sample_weights.txt'%path,sample_weights)

print('written predictions')

for i in history.history.keys() :
    np.savetxt('%s/%s.txt'%(path,i),history.history[i])
print('history finished')

model_yaml = model.to_yaml()
with open("%s/model.yaml"%path, "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("%s/model.h5"%path)
print("Saved model to disk")


