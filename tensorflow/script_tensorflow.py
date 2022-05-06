#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from decimal import Decimal
import numpy as np
import glob
import importlib
import gc
import matplotlib.pyplot as plt
import joblib
import datetime, os
#from livelossplot import PlotLossesKeras
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load
from numpy import savez_compressed

from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, TimeDistributed, ConvLSTM2D, Input
from keras.layers.core import Dense, Flatten, Dropout, RepeatVector, Reshape
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, Conv1D
from keras.callbacks import ModelCheckpoint

from platform import python_version
import keras
import tensorflow as tf
#get_ipython().run_line_magic('load_ext', 'tensorboard')
from keras.callbacks import TensorBoard
import time
import math

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print('Notebook running on Python', python_version())
print('Numpy version', np.version.version)
print('Scikit-learn version {}'.format(sklearn.__version__))
print('Keras version ', keras.__version__,'and TensorFlow', tf.__version__, '(CUDA:', tf.test.is_built_with_cuda(), '- GPUs available:', get_available_gpus(), ')')


# In[ ]:


#get_ipython().system('nvidia-smi')


# In[ ]:


n_channels = 3
n_input = 3
n_output = 1
leakyrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)


# ## Create the data generator

# In[ ]:


# list_datasets shape -> (cases, samples per case, dim X, dim Y, dim Z, channels)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_datasets, batch_size=128, dim=512, shuffle=True, n_channels=3, observation_samples=3):
        #print('Generator Initialization')
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.n_input = observation_samples      
        self.n_output = 1 #not implemented yet !!!
        self.last_samples = self.n_input - 1 + self.n_output      
        
        self.indexes = []
        cnt = 0
        self.total_samples = 0
        for case in list_datasets:
            samples = len(case) - self.last_samples
            self.total_samples += samples
            for sample in range(samples):
                self.indexes.append(cnt)
                cnt += 1
            cnt += self.last_samples
        
        self.ds = np.concatenate((list_datasets), axis = 0)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.total_samples / self.batch_size))
  
    def __getitem__(self, batch_index):          
        'Generate one batch of data through dataset'
        list_IDs_temp = self.indexes[batch_index * self.batch_size : (batch_index+1) * self.batch_size]
        #print(index, list_IDs_temp)
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #print('Yieded batch %d' % index)
        return X, y

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dim * self.n_input,  self.n_channels))
        y = np.empty((self.batch_size, self.dim * self.n_output, self.n_channels))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i] = np.concatenate((self.ds[ID:ID+self.n_input]), axis = 0)
            y[i] = self.ds[ID+self.n_input]  
            
        X = np.reshape(X, (X.shape[0], self.n_input, -1))
        y = np.reshape(y, (y.shape[0], self.n_output, -1))

        return X, y


# In[ ]:


basepath = "/path/to/rubbish"
datasetpath = basepath + "/NPZs/"
modelpath = "/path/to/dataset"
if not os.path.exists(modelpath):
    os.mkdir(modelpath)

dsaux = np.load(modelpath + "/dataset.npz")
dsaux = dsaux.f.data
print(dsaux.shape)
samples_per_case = np.load(modelpath + "/samples.npz")
samples_per_case = samples_per_case.f.data.tolist()
print(len(samples_per_case))
dsaux = dsaux.reshape(dsaux.shape[0], -1, n_channels)
print(dsaux.shape)
n_cells = dsaux.shape[1]
acc = 0
ds_ready = []
for case in range(len(samples_per_case)):
    idx_ini = acc
    acc += samples_per_case[case]
    ds_ready.append(dsaux[idx_ini:acc, :])
ds_ready=np.array(ds_ready)
print(ds_ready.shape)
#print(ds_ready[0].shape)
print('Dataset size in memory: %0.2f GB' % (ds_ready.nbytes / 1024**3)) 
np.info(ds_ready)


n_files = len(ds_ready)
valTo = int(n_files * 0.8)
trainTo = valTo - int(valTo * 0.2)
valFrom = trainTo

testFrom = valTo
testTo = n_files

print("Train from 0 to %d" % (trainTo))
print("Validation from %d to %d" % (valFrom, valTo))
print("Test from %d to %d" % (testFrom, testTo))

print(ds_ready[:trainTo].shape)
print(ds_ready[valFrom:valTo].shape)
print(ds_ready[testFrom:testTo].shape)



# In[ ]:


files = glob.glob(datasetpath + "/*.npz")
print(files[-27:][19])


# In[ ]:


n_epochs=10000
n_batch = 14
train = DataGenerator(ds_ready[:trainTo], 
                      batch_size=n_batch, dim=n_cells, shuffle=True, observation_samples=n_input, n_channels = 3)
validation = DataGenerator(ds_ready[valFrom:valTo], 
                      batch_size=n_batch, dim=n_cells, shuffle=True, observation_samples=n_input, n_channels = 3)


# In[ ]:


x, y = train[0]
print(x.shape)
print(y.shape)


# ## Define the ML model

# In[16]:


n_timesteps = n_input
n_features = n_cells * n_channels
n_outputs = n_output


# In[17]:


opt = tf.keras.optimizers.Adam(learning_rate=0.00025)
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=False))
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(n_features)))

model.compile(loss='mae', optimizer=opt)
model.summary()


# ## Training

# In[18]:

start = time.time()
H = model.fit(train, validation_data=validation, steps_per_epoch=math.ceil(2189.0), epochs=20, verbose=1, shuffle=True)
end = time.time()
print("Total training time", end - start)
