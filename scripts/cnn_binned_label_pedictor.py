#!/usr/bin/env python
# coding: utf-8
import errno
import tensorflow as tf
import os
import os.path
import numpy as np
import skimage.measure
import time


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Activation,Dropout,MaxPooling1D,AveragePooling1D,BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_yaml
import yaml
from tensorflow.python.keras.layers import deserialize


#Using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3))
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
path = './../models/cnn_model_classification__lay_6__ksize_5__psize_3__dilrate_1__dropout_2__0.25__binsize__20__epochs_10__bs_128__train_28k__val_2k___generator_inp_size_1281_test_pos_enc_t'
mkdir_p(path)
print('path created')

num_train = 90000
num_val   = 5000
num_test  = 5000

bin_size   = 4 #In this model, the data is already binned to size 5k from original spectra of size 25k. This is additional binning.
input_size= 5121//bin_size  
num_output= input_size  #Number of outputs can be set as per choice. For this model, we are classifying each point. Hence, number of outputs are equal to number of points in input 
num_class = 4
batch_size = 128
steps_per_epoch = 200 #number of training steps
validation_steps = 10 #number of validation steps
pos_enc  = True
num_epochs     = 10
initial_epoch  = 0
num_batchsize  = 128

'''
Loading 100k examples of spectrum data and mode data (label data)
Dictionary
    0: mode l=0
    1: mode l=1
    2: mode l=2
    3: noise
'''

start_time=time.time()
labels    = np.load('./../binned_data/binned_labels_100k.npy') #shape=(examples,labels of each spectra=5121)
end_time=time.time()
print('Time taken for loading mode data= %f s'%(end_time-start_time))

print('loaded mode data and its shape is (%d,%d)'%(labels.shape[0],labels.shape[1]))

#Reduces the labels taking the bin_size number of points to 1 point defined by function in func. Ignores the last label.
labels = skimage.measure.block_reduce(labels,block_size=(1,bin_size),func=np.min)[:,:-1]
labels = labels.astype(np.int32)

print('Binned label data and its shape is (%d,%d)'%(labels.shape[0],labels.shape[1]))

#Reduces the spectrum data taking the bin_size number of points to 1 point defined by function in func. Ignores the last datapoint.
start_time=time.time()
spectrum_data = np.load('./../binned_data/binned_data_100k.npy') #shape=(examples,length of spectrum)
end_time=time.time()
print('Time taken for loading spectrum data= %f s'%(end_time-start_time))

print('loaded spectrum data and its shape is (%d,%d)'%(spectrum_data.shape[0],spectrum_data.shape[1]))

spectrum_data = skimage.measure.block_reduce(spectrum_data,block_size=(1,bin_size),func=np.mean)[:,:-1]
spectrum_data = spectrum_data.reshape(spectrum_data.shape[0],spectrum_data.shape[1],1)

print('Binned spectra data and its shape is (%d,%d,1)'%(spectrum_data.shape[0],spectrum_data.shape[1]))


print('loaded data')


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



if pos_enc ==True:
    spectrum_data = positional_enc(spectrum_data)


# Creaating X,Y

X_train=spectrum_data[:num_train]
y_train=labels[:num_train]
y_train= y_train.T # Reshaping labels array to (num_outputs,num_examples)


X_val = spectrum_data[num_train:num_train+num_val]
y_val = labels[num_train:num_train+num_val]
y_val = y_val.T # Reshaping labels array to (num_outputs,num_examples)


del spectrum_data
del labels

print('data preparation completed')

def traingenerator(X_train, y_train, num_batchsize):
    '''
    yields training data with given batchsize. 
    It takes input as X_train and Y_train and yields
    batches to fit generator for training.
    '''
    
    num_train = X_train.shape[0]

    while 1:
        for i in range(num_train//num_batchsize): 
            z_train = y_train[:,i*num_batchsize:(i+1)*num_batchsize]        
            print(i)
            yield X_train[i*num_batchsize:(i+1)*num_batchsize], list(z_train)

def valgenerator(X_val, y_val, num_batchsize):
    '''
    yields validation data with given num_batchsize. 
    It takes input as X_val and Y_val and yields
    batches to fit generator for validation.
    '''
    
    num_val = X_val.shape[0]
    
    while 1:
        for i in range(num_val//num_batchsize):
            z_train = y_train[:,i*num_batchsize:(i+1)*num_batchsize]
            print(i)
            yield X_train[i*num_batchsize:(i+1)*num_batchsize], list(z_train)
            
        

def scheduler(epoch):
    if epoch <= 20:
        return 0.001
    elif epoch<=40:
        return 0.0008
    else:
        return 0.0005





def cnn_model(input_size, num_output, filters, kernel_size, strides, pool_size, dilation_rate, pos_enc=False):
    '''
    Returns a CNN model for given 
    input_shape, num_output, filters (first layer filters)
    kernel_size, pool_size, dilation_rate, pos_enc 

    filters get doubled after each layer in this model

    pos_enc: Positional encoding is set to be false. If True, 
    then the input shape will change accordingly.
    '''
    visible = Input(shape=(input_size,1))
    if pos_enc==True:
        visible = Input(shape=(input_size,2))

    x = visible
    for j in range(0,6):    #i: Num layers/pool, j:num components of i
        for i in range(0,1):
            x =  Conv1D(filters = filters,kernel_size = kernel_size, dilation_rate=dilation_rate, strides=strides, padding='same', activation='relu')(x)
            x =  BatchNormalization()(x)
            filters = 2*filters
        x = MaxPooling1D(pool_size = pool_size, strides=strides, padding='same')(x)
        x = AveragePooling1D(pool_size=pool_size, padding='same')(x)
        if j>=4:
            x = Dropout(0.25)(x)

    flat = Flatten(name='flatten')(x)

    output=[]
    for i in range(1,num_output+1):
        name = f'output_{i}'
        vars()["output"+str(i)]= Dense(4, activation='softmax',name=name)(flat)#,name='output'+str(i)
        output.append(vars()["output"+str(i)])

    model = Model(inputs=visible, outputs=output)
    return model



model = cnn_model(input_size, num_output,kernel_size = 5,strides = 1,pool_size = 3,filters = 16,dilation_rate = 1, pos_enc=pos_enc)

'''
# Script to load a model and load weights
yaml_file = open('%s/model.yaml'%path, 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
config = yaml.load(loaded_model_yaml,Loader=yaml.UnsafeLoader)
model = deserialize(config)
# load weights into new model
model.load_weights("%s/model.h5"%path)
print("Loaded model from disk")
#Change initial_epoch to last epoch number
initial_epoch = 10 #change this appropriately to last epoch in already trained model
num_epochs = 20 # Change appropriately to total number of epochs to train (already trained + extra epochs) (for eg. 10 already +10 now)
'''

print(model.summary())

print('compiltion started')

#Assigning Class weights (approximately)
class_weight = {0: 0.8, 1: 0.6, 2: 0.6, 3: 0.15}
class_weights = {}
metrics_array = {}
loss_array = {}
for i in range(1,num_output+1):
    name = f'output_{i}'
    metrics_array[name] = 'sparse_categorical_accuracy'
    loss_array[name] = 'sparse_categorical_crossentropy'
    class_weights[name] = class_weight

model.compile(loss=loss_array,optimizer=tf.keras.optimizers.Adam(),metrics=metrics_array)

checkpoint = ModelCheckpoint(filepath= path+'/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',period=2)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

print('Writing model')

model_yaml = model.to_yaml()
with open("%s/model.yaml"%path, "w") as yaml_file:
    yaml_file.write(model_yaml)

print('Training started')

start_time=time.time()
history = model.fit(traingenerator(X_train,y_train,num_batchsize), validation_data=valgenerator(X_val,y_val,num_batchsize), steps_per_epoch = steps_per_epoch, validation_steps=validation_steps, epochs = num_epochs, initial_epoch=initial_epoch, verbose=2,class_weight=class_weights, callbacks=[callback,checkpoint]) 
end_time=time.time()
print('Time taken for training model= %f s'%(end_time-start_time))


print('saving weights')
model_yaml = model.to_yaml()
with open("%s/model.yaml"%path, "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("%s/model.h5"%path)
print("Saved model to disk")

print('Saving loss,val_loss,lr among the history of the model. There are several other items in history like output_1_sparse_categorical_accuracy etc., printing history.history.keys() shows you all the items.')
#print(history.history.keys())
for i in ['loss','val_loss','lr'] :
    np.savetxt('%s/%s.txt'%(path,i),history.history[i])
print('history finished')

