# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:42:51 2019

@author: Eric Born
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM,\
                                    BatchNormalization, Flatten, GRU, Input,\
                                    Embedding
                                    
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint,\
                                              TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import RMSprop
#import keras
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import collections
import pandas as pd
import numpy as np
import os
import random
import time

# removes scientific notation from np prints, prints numbers as floats
np.set_printoptions(suppress=True)

###!!! Maybe only keep matches where the bot won, indicating good decision making?

path = 'C:/botdata/'

# list of all files
files = os.listdir(path)

# object for single array file, which equals a single game
full_array = np.load(path + files[0], allow_pickle=True)

# setup x and y
# x will be the supply_data inputs
# y will be the action that was taken

# array that contains supply data, all rows, index 0-14
# 15 inputs
#supply_array = full_array[:,0:15]
x_data = full_array[:,0:15]

# setup an array that only includes the action data index 15-23
# 9 outputs
#actions_array = full_array[:,15:24]
y_data = full_array[:,15:24]

# 427 by 15 and by 9
x_data.shape
y_data.shape

# setup number to split for train/test
num_train = int(0.9 * len(x_data))

# split x train/test
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)

# split y train/test
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)

# input columns
num_x_signals = x_data.shape[1]
num_x_signals

# output columns
num_y_signals = y_data.shape[1]
num_y_signals

# Setup scalers for x and y
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.fit_transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.fit_transform(y_test)

# check shapes for the scaled sets
x_train_scaled.shape
y_train_scaled.shape

# create a batch generator to feed the data into the NN
def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)

# larger batch size the more is fed to the GPU at once, adjust if there
# are memory issues
batch_size = 6

# TODO
# This probably needs to be variable to account for different number of steps
# per game due to variable game length
# length of steps in the first game
sequence_length = 12

# create a generator object
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

x_batch, y_batch = next(generator)


validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

# start building the NN model
# Sequential allows the model to be build one layer at a time
# with each subsequent layer being added to the first
model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))

# The GRU outputs a batch of sequences of 512 values. 
# We want to predict 9 output-signals, so we add a fully-connected (or dense) 
# layer which maps 512 values down to only 9 values.
model.add(Dense(num_y_signals, activation='sigmoid'))
# A problem with using the Sigmoid activation function, is that we can now only
# output values in the same range as the training-data.
# If new numbers in the training data are higher or lower
if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))

warmup_steps = 50

def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)
print(model.summary())

# build writing checkpoints
path_checkpoint = 'GRU_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

# set early stop if the performance worsens on validation
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

# setup tensorboard
callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

# setup automatic reduction of learning rate
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-8,
                                       patience=0,
                                       verbose=1)

# setup list to hold all callback info
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

timings_list = []

# start timing
timings_list.append(['Model start:', time.time()])

model.fit_generator(generator=generator,
                    epochs=20,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)

timings_list.append(['Model end:', time.time()])
