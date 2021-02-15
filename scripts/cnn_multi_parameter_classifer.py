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

num_train      = 1240000#1240000#820000#910000#210000#980000#340000#990000#590000#170000#650000
num_test       = 2000#2000#4000#4000#4000#5000#5000#8000#30000#2000#30000
num_validation = 50000#50000#40000#40000#10000#40000#16000#40000#40000#4000#50000
examples       = num_train + num_validation + num_test

num_classes_dnu = 55
num_classes_dp  = 45
num_classes_q   = 25
num_classes_acr = 30
num_classes_aer = 17
num_classes_inc = 18
num_classes_epp = 20
num_classes_epg = 10

num_epochs     = 50#25
num_batchsize  = 128

loss           = 'sparse_categorical_crossentropy'
pos_enc        = False #False/True

path = './../models/multi_parameter_classifier/cnn_lstm_model__params_Dnu_Dp_q_epp_epg_acr_aer_inc__pos_enc_f__cnn_lay_6_lstm_lay_2_fil_256_dlay_1_n_200__drop_2_0.25__ksize_5_filters_16__psize_3__dil_rate_1__epochs_50__bs_128__train_eg_1250k__activ_relu_dnu_4_9_data_2_lr_3'


mkdir_p(path)
print('path created')

#loading X data
start_time=time.time()
print(start_time)
X = np.load('/mnt/sdc/siddharth/spectrum_data_dnu_4_10.npy')
end_time=time.time()
print(X.shape)
print(end_time)
print('Time taken for loading data= %f s'%(end_time-start_time))


Y = np.loadtxt('/mnt/sdc/siddharth/data_y_dnu_4_10_unnormalized_dp_lim.npy')
inc = np.load('/mnt/sdc/siddharth/data_inc_dnu_4_10_unnormalized_dp_small.npy')

print(X.shape,Y.shape)

np.random.seed(0)
np.random.shuffle(X)

np.random.seed(0)
np.random.shuffle(Y)

np.random.seed(0)
np.random.shuffle(inc)

print('dataset preparation completed')
# 5 parameters among these Y. (average core rotation, average env rot, Delta nu, Delta pi, coupling factor q)
aer, acr, Dnu, Dp, epsilon_g ,q = Y[:,0], Y[:,1], Y[:,2], Y[:,7], Y[:,8], Y[:,9] 

epsilon_p = Y[:,3] 

snr = Y[:,10]

sample_weights = np.ones(Dnu.shape)
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

def categorize(par,num_cl):
    par = par*num_cl
    par = np.floor(par)
    par = par.astype('int')
    par = np.where(par<num_cl,par,num_cl-1)
    return par


Dnu = categorize(Dnu,num_classes_dnu)
Dp  = categorize(Dp,num_classes_dp)
q   = categorize(q,num_classes_q)
acr = categorize(acr,num_classes_acr)
aer = categorize(aer,num_classes_aer)
inc = categorize(inc,num_classes_inc)
epsilon_p = categorize(epsilon_p,num_classes_epp)
epsilon_g = categorize(epsilon_g,num_classes_epg)


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

X1_train = X1[:num_train]
X1_val   = X1[num_train:num_train+num_validation]
X1_test  = X1[num_train+num_validation:]

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

epsilon_p_train = epsilon_p[:num_train]
epsilon_p_val   = epsilon_p[num_train:num_train+num_validation]
epsilon_p_test  = epsilon_p[num_train+num_validation:]

epsilon_g_train = epsilon_g[:num_train]
epsilon_g_val   = epsilon_g[num_train:num_train+num_validation]
epsilon_g_test  = epsilon_g[num_train+num_validation:]

inc_train = inc[:num_train]
inc_val   = inc[num_train:num_train+num_validation]
inc_test  = inc[num_train+num_validation:]

sample_weights_train = sample_weights[:num_train]
sample_weights_val   = sample_weights[num_train:num_train+num_validation]

#This list depends on the number of parameters
string = ['dnu','dp','q','acr','aer','inc','epp','epg']
labels_train = [Dnu_train,Dp_train,q_train,acr_train,aer_train,inc_train,epsilon_p_train,epsilon_g_train]#[Dp_train,Dnu_train,q_train,aer_train,acr_train,epsilon_p_train,epsilon_g_train]
labels_val   = [Dnu_val,Dp_val,q_val,acr_val,aer_val,inc_val,epsilon_p_val,epsilon_g_val]#[Dp_val,Dnu_val,q_val,aer_val,acr_val,epsilon_p_val,epsilon_g_val]


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

def cnn_model(X,num_classes_dnu,num_classes_dp,num_classes_q,num_classes_acr,num_classes_aer,num_classes_inc,num_classes_epp,num_classes_epg,pos_enc=False):
    kernel_size = 5
    strides = 1
    pool_size = 3
    filters = 16
    dilation_rate=1

    visible = Input(shape=(X.shape[1],1))

    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible


    for j in range(0,6):    #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  Conv1D(filters = filters,kernel_size = kernel_size, dilation_rate=dilation_rate, strides=1, padding='same', activation='relu')(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        x = MaxPooling1D(pool_size = pool_size, strides=1, padding='same')(x)
        x = AveragePooling1D(pool_size=pool_size, padding='same')(x)
        if j>=4:
            x = Dropout(0.25)(x)
    

    x = LSTM(256,return_sequences=True)(x)
    x = Dropout(0.25)(x)
   
    x = LSTM(512,return_sequences=True)(x)
    x = Dropout(0.25)(x)
    flat = Flatten(name='flatten')(x)


    flat = Dense(200, activation='tanh')(flat)
    flat = Dropout(0.25)(flat)


    output1 = Dense(num_classes_dnu, activation='softmax', name='Dnu')(flat)
    output2 = Dense(num_classes_dp , activation='softmax', name='Dp')(flat)
    output3 = Dense(num_classes_q  , activation='softmax', name='q')(flat)
    output4 = Dense(num_classes_acr, activation='softmax', name='acr')(flat)
    output5 = Dense(num_classes_aer, activation='softmax', name='aer')(flat)
    output6 = Dense(num_classes_inc, activation='softmax', name='inc')(flat)
    output7 = Dense(num_classes_epp, activation='softmax', name='epsilon_p')(flat)
    output8 = Dense(num_classes_epg, activation='softmax', name='epsilon_g')(flat)

    output  = [output1,output2,output3,output4,output5,output6,output7,output8] #Add additional outputs/parameters according to requirements

    model = Model(inputs=visible, outputs=output)
    return model


def cnn_attention_model(X,num_classes,pos_enc=False):
    kernel_size = 5
    strides = 1
    pool_size = 3
    filters = 16
    dilation_rate=1

    visible = Input(shape=(X.shape[1],1))

    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible
    #b = AveragePooling1D(pool_size=5, padding='same')(x)
    #b = tf.keras.backend.squeeze(b, axis=-1)

    for j in range(0,6):    #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  Conv1D(filters = filters,kernel_size = kernel_size, dilation_rate=dilation_rate, strides=1, padding='same', activation='relu')(x)
            x =  BatchNormalization()(x)

            #x = LSTM(filters, return_sequences=True)(x)
            #x =  BatchNormalization()(x)
            filters = 2*filters
        x = MaxPooling1D(pool_size = pool_size, strides=1, padding='same')(x)
        x = AveragePooling1D(pool_size=pool_size, padding='same')(x)
        #x  = tf.keras.layers.Attention(causal=False)([x,x])
        if j>=4:
            x = Dropout(0.25)(x)
    #x = GRU(256, return_sequences=True)(x)
    #x = Dropout(0.25)(x)
    #x = GRU(512, return_sequences=True)(x)
    #x = Dropout(0.25)(x)
    #x = tf.keras.layers.concatenate([visible,x], axis=-1)
    #b = LSTM(64, return_sequences=True)(b)

    x  = tf.keras.layers.AdditiveAttention(causal=False)([x,x])

    x = LSTM(256, dropout=0.2 ,recurrent_dropout=0.2 ,return_sequences=True)(x)#Bidirectional(LSTM(256, return_sequences=True))(x)
    #x = tf.keras.layers.concatenate([b,x], axis=-2)
    #x = Dropout(0.25)(x)
   
    x = LSTM(512,dropout=0.2 ,recurrent_dropout=0.2 ,return_sequences=True)(x)#Bidirectional(LSTM(512, return_sequences=True))(x)



    #x = Dropout(0.25)(x)
    #x = LSTM(1024, return_sequences=True)(x)
    #x = Dropout(0.25)(x)
    #x = LSTM(2048, return_sequences=True)(x)
    #x = Dropout(0.25)(x)
    flat = Flatten(name='flatten')(x)
    #flat = tf.keras.layers.concatenate([flat,b], axis=-1)

    flat = Dense(500, activation='tanh')(flat)
    #flat = Dense(500, activation='tanh')(flat)
    flat = Dropout(0.25)(flat)

    output1 = Dense(num_classes, activation='softmax', name='parameter')(flat)
    #output2 = Dense(1, activation='linear', name='Dnu')(flat)
    #output3 = Dense(1, activation='linear', name='q')(flat)
    #output4 = Dense(1, activation='linear', name='aer')(flat)
    #output5 = Dense(1, activation='linear', name='acr')(flat)
    #output6 = Dense(1, activation='linear', name='epsilon_p')(flat)
    #output7 = Dense(1, activation='linear', name='epsilon_g')(flat)

    output = output1
    #output  = [output1,output2,output3,output4,output5,output6,output7] #Add additional outputs/parameters according to requirements

    model = Model(inputs=visible, outputs=output)
    #plot_model(model, to_file='%s/model.png'%path,show_shapes=True)
    #print(model.summary())
    return model


def cnn_refined_model(X,X1,num_classes,pos_enc=False):
    kernel_size = 5
    strides = 1
    pool_size = 3
    filters = 16
    dilation_rate=1

    visible = Input(shape=(X.shape[1],1))
    visible1 = Input(shape=(X1.shape[1],))


    if pos_enc==True:
        visible = Input(shape=(X.shape[1],2))

    x = visible
    for j in range(0,6):    #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  Conv1D(filters = filters,kernel_size = kernel_size, dilation_rate=dilation_rate, strides=1, padding='same', activation='relu')(x)
            #x = LSTM(filters, return_sequences=True)(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        x = MaxPooling1D(pool_size = pool_size, strides=1, padding='same')(x)
        x = AveragePooling1D(pool_size=pool_size, padding='same')(x)
        if j>=4:
            x = Dropout(0.25)(x)
    x = LSTM(32, return_sequences=True)(x)
    #x = Dropout(0.25)(x)

    flat = Flatten(name='flatten')(x)
    flat = tf.keras.layers.concatenate([flat,visible1], axis=-1)

    output1 = Dense(num_classes, activation='softmax', name='parameter')(flat)
    output2 = Dense(1, activation='linear', name='Dnu')(flat)
    output3 = Dense(1, activation='linear', name='q')(flat)
    #output4 = Dense(1, activation='linear', name='aer')(flat)
    #output5 = Dense(1, activation='linear', name='acr')(flat)
    #output6 = Dense(1, activation='linear', name='epsilon_p')(flat)
    #output7 = Dense(1, activation='linear', name='epsilon_g')(flat)

    output = output1
    #output  = [output1,output2,output3,output4,output5,output6,output7] #Add additional outputs/parameters according to requirements

    model = Model(inputs=[visible,visible1], outputs=output)
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
    for j in range(0,7):  #i: Num layers/pool, j:num components of i
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


#model = cnn_refined_model(X,X1,num_classes,pos_enc)
model = cnn_model(X,num_classes_dnu,num_classes_dp,num_classes_q,num_classes_acr,num_classes_aer,num_classes_inc,num_classes_epp,num_classes_epg,pos_enc)#cnn_model(X,num_classes_dnu,num_classes_dp,num_classes_q,pos_enc)#cnn_model(X,num_classes,pos_enc)

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
    if epoch <= 8:#5: #2
        return 0.0003
    elif epoch<= 15:#8:
        return 0.00018#0
    elif epoch<=25:#15: 
        return 0.0001
    elif epoch<=40:#25:
        return 0.00005
    else:
        return 0.00001 #5

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

checkpoint = ModelCheckpoint(filepath= path+'/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',period=5)

callbacks = [callback,checkpoint]
model_yaml = model.to_yaml()
with open("%s/model.yaml"%path, "w") as yaml_file:
    yaml_file.write(model_yaml)

#Compiling model
loss = [loss]*len(string)
metrics= ['sparse_categorical_accuracy']*len(string)
model.compile(loss=loss,optimizer=tf.keras.optimizers.Adam(),metrics=metrics)

start_time = time.time()
history=model.fit(X_train, labels_train, epochs=num_epochs,validation_data=(X_val, labels_val), batch_size=num_batchsize,initial_epoch=0, verbose=1,callbacks=callbacks)
end_time = time.time()
print('Time taken for training= %f s'%(end_time-start_time))

#Writing predictions
predictions_items = model.predict(X)
for i in range(len(string)):
    predictions = predictions_items[i]
    predictions = np.array(predictions)
    predictions = np.squeeze(predictions)
    predictions = predictions.T
    np.save('%s/predicted_parameters_%s.npy'%(path,string[i]),predictions)

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


