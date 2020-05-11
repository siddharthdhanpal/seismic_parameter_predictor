#!/usr/bin/env python
# coding: utf-8
import errno
import tensorflow as tf
import os
import os.path
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Activation,Dropout,MaxPooling1D,AveragePooling1D,BatchNormalization
from tensorflow.keras.layers import Input,Conv1D,SeparableConv1D
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

import time
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint


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

num_train      = 26000
num_test       = 2000
num_validation = 2000
examples       = num_train + num_validation + num_test

num_epochs     = 70
num_batchsize  = 128
#lr             = 0.00005 # Now, lr is defined as scheduler before fitting
loss           = 'mse'
pos_enc        = False #False/True

path = './../models/sep_cnn_model__params_5_dp_dnu_q_aer_acr__pos_enc_f__lay_6__drop_1_0.25__ksize_5__filters_16__str_psize_5__loss_mse__epochs_70__bs_128__train_eg_260k__activ_tanh'


mkdir_p(path)
print('path created')

#loading X data
start_time=time.time()
print(start_time)
X=np.load('./../data/data_30k.npy')#shape = (examples,length of spectrum)-(30000,25606)
end_time=time.time()
print(X.shape)
print(end_time)
print('Time taken for loading data= %f s'%(end_time-start_time))

Y = np.loadtxt('./../data/labels_30k.txt') #shape = (examples,parameters) -  Y has many parameters - (30000,num_parameters)

# 5 parameters among these Y. (average core rotation, average env rot, Delta nu, Delta pi, coupling factor q)
aer, acr, Dnu, Dp, q = Y[:,0], Y[:,1], Y[:,2], Y[:,7], Y[:,9] 

#epsilon_p = Y[:,3] 


#Scale parameters in between [0,1]
def scale_parameter(a):
    a = (a-a.min())/(a.max()-a.min())
    return a

aer = scale_parameter(aer)
acr = scale_parameter(acr)
Dnu = scale_parameter(Dnu)
Dp  = scale_parameter(Dp)
q   = scale_parameter(q)

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

aer_train = aer[:num_train]
aer_val   = aer[num_train:num_train+num_validation]
aer_test  = aer[num_train+num_validation:]

acr_train = acr[:num_train]
acr_val   = acr[num_train:num_train+num_validation]
acr_test  = acr[num_train+num_validation:]

Dnu_train = Dnu[:num_train]
Dnu_val   = Dnu[num_train:num_train+num_validation]
Dnu_test  = Dnu[num_train+num_validation:]

Dp_train = Dp[:num_train]
Dp_val   = Dp[num_train:num_train+num_validation]
Dp_test  = Dp[num_train+num_validation:]

q_train = q[:num_train]
q_val   = q[num_train:num_train+num_validation]
q_test  = q[num_train+num_validation:]

#This list depends on the number of parameters
labels_train = [Dp_train,Dnu_train,q_train,aer_train,acr_train]
labels_val   = [Dp_val,Dnu_val,q_val,aer_val,acr_val]


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
#CNN Model

def cnn_model(X,pos_enc=False):
    kernel_size = 5
    strides = 1
    pool_size = 5
    filters = 16

    visible = Input(shape=(X.shape[1],1))
    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible
    for j in range(0,6):    #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  Conv1D(filters = filters,kernel_size = kernel_size, strides=1, padding='same', activation='tanh')(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        x = MaxPooling1D(pool_size = pool_size, strides=1, padding='same')(x)
        x = AveragePooling1D(pool_size=pool_size, padding='same')(x)
        if j>=5:
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
    output  = [output1,output2,output3,output4,output5] #Add additional outputs/parameters according to requirements

    model = Model(inputs=visible, outputs=output)
    #plot_model(model, to_file='%s/model.png'%path,show_shapes=True)
    #print(model.summary())
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
        if j>=5:
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
            x =  Conv1D(filters = filters,kernel_size = kernel_size, strides=1, padding='same', activation='tanh')(x)
            x =  BatchNormalization()(x)
            x =  Conv1D(filters = filters,kernel_size = kernel_size, strides=stride_pool, padding='same', activation='tanh')(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        if j>=5:
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
        if j>=5:
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


model = pure_separable_cnn_model(X,pos_enc)
print(model.summary())


def scheduler(epoch):
    if epoch <= 10:
        return 0.001
    elif epoch<=20:
        return 0.0001
    else:
        return 0.00005
    
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


checkpoint = ModelCheckpoint(filepath= path+'/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')

model_yaml = model.to_yaml()
with open("%s/model.yaml"%path, "w") as yaml_file:
    yaml_file.write(model_yaml)

#Compiling model
model.compile(loss=loss,optimizer=tf.keras.optimizers.Adam(),metrics=[pearson_correlation_coefficient,r2,explained_variance])

start_time = time.time()
history=model.fit(X_train, labels_train, epochs=num_epochs,validation_data=(X_val, labels_val), batch_size=num_batchsize, verbose=1,callbacks=[callback,checkpoint])
end_time = time.time()
print('Time taken for training= %f s'%(end_time-start_time))

#Writing predictions
predictions = model.predict(X)
predictions = np.array(predictions)
predictions = np.squeeze(predictions)
predictions = predictions.T
np.savetxt('%s/predicted_parameters.txt'%path,predictions)
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
