# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:19:22 2019

@author: Eric Born
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM,\
                                    BatchNormalization, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
#import keras
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Other methods
###################################
###!!! First layer should be supply_data array, 13 neurons wide
###!!! ??Second layer could be the actions_data array, 9 neurons wide??
###!!! ??Another layer could be the troop_data array 6 neurons wide??

###!!! Could also be single input layer that is comprised of all three metrics 28 wide


###!!! Output layer should be actions_data choices 9 neurons wide


# Tests if tf is up and ready
#tf.test.is_built_with_cuda()
#
## Tests if GPU is ready
#tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# column list for all match data
#cols = ['supply_cap', 'supply_army', 'supply_workers', 'nexus', 'pylons', 
#        'assimilators', 'gateways', 'cybercore', 'robofac', 'stargate', 
#        'robobay', 'd-structures', 'd-units',  'attack', 'assimilators', 
#        'offensive_force', 'pylons', 'workers', 'distribute', 'nothing', 
#        'expand buildings', 'ZEALOT', 'STALKER', 'ADEPT', 'IMMORTAL', 
#        'VOIDRAY', 'COLOSSUS', 'difficulty', 'outcome']

path = 'C:/botdata/'

# list of all files
files = os.listdir(path)

# object for single array file, which equals a single game
full_array = np.load(path + files[0], allow_pickle=True)

# array that contains supply data, all rows, index 0-14
# 15 inputs
supply_array = full_array[:,0:15]

# setup an array that only includes the action data index 15-23
# 9 outputs
actions_array = full_array[:,15:24]

# Setup scalers for supply_array
scaler = StandardScaler()
scaler.fit(supply_array)
supply_array_scaled = scaler.transform(supply_array)


# 80% train 20% test
(supply_array_scaled_train_x, supply_array_scaled_test_x, 
 supply_array_scaled_train_y, supply_array_scaled_test_y) = (
        train_test_split(supply_array_scaled, actions_array, test_size = 0.20, 
                         random_state=1337))

# one batch of data, full length of dataset
seq_length = supply_array_scaled_train_x.shape[0]



# outcome isnt being used current, maybe save for reward function?
# outcome -1 loss, 0 tie, 1 win
# create an array the length of the current full_array with all 0's
#sc_y = np.zeros(len(full_array))

# fill the array with 1 for win, -1 for loss
# outcome is only recorded in the last frame of the game, last index, 
# hence [-1][-1]
#if full_array[-1][-1] == 1:
#    sc_y.fill(1)
#elif full_array[-1][-1] == -1:
#    sc_y.fill(-1)


# Create an empty list to hold all actions
#action_list = []
#
## store all action data into action list
#for i in range(0, len(a)):
#    action_list.append(a[i][15:24])
#    np.delete(a_test[i][15:24])
#    

# View the data in various ways
## all rows from single game, supply_data
#for i in range(0, len(full_array)):
#    print(full_array[i][0:15])
#
## all rows from single game, action_data
#for i in range(0, len(full_array)):
    #print(full_array[i][15:24])
    
## all rows from single game, troop_data
#for i in range(0, len(full_array)):
#    print(full_array[i][24:30])
#
## all rows from single game, outcome_data
#for i in range(0, len(full_array)):
#    print(full_array[i][30:32])
#
## represents a single step from a game, all 4 data arrays
#a[0]
#
## representsa single step, supply_data
#a[0][0:15]
#
## single step, action_data
#a[0][15:24]
#
## single step, troop_data
#a[0][24:30]
#
## single step, outcome_data
#a[0][30:32]

# scaled data
#train_scaled = preprocessing.scale(train)
#test_scaled = preprocessing.scale(test)
#a_scaled = preprocessing.scale(a)
#
#a_scaled[0][0:15]

# create sequences for the data
#sequential_data = []
#prev_frames = collections.deque(maxlen=SEQ_LEN)
#
#for i in a_scaled:  # iterate over the values
#    prev_frames.append([n for n in i[:-1]])  # store all but the target
#    if len(prev_frames) == SEQ_LEN:  # make sure we have 60 sequences!
#        sequential_data.append([np.array(prev_frames), i[-1]])  # append those bad boys!
#
## Shuffle the data to prevent memorization instead of learning   
#random.shuffle(sequential_data)
#
## find the index for the 95th %
#last_5pct = -int(0.05*len(a))  
#
## first 95% of data and only suppy_data
#main = a[0:last_5pct][0:15]
#
## last 5% of data and only suppy_data
#validation = a[last_5pct:len(a)][0:15]







# define empty dataframe using columns previously defined
#full_df = pd.DataFrame(columns=cols)
#
## for loop that evaluates the numpy files in index 11 to look for match outcome
## 1 means win, 0 means tie, -1 means loss
#for file in range(0, len(files)):
#    #full_df.loc[file] = np.load(path + files[file], allow_pickle=True)
#    a = np.load(path + files[file], allow_pickle=True)
#    for i in range(0, len(a)):
#        supply_data.append(a[i][0])
#    
    # reads last array from each file filted down to the outcome data
    #print(np.load(path + files[file], allow_pickle=True)[:][-1][-1])

# creates labels and data sets
#labels = full_df['outcome']
#data = full_df
#
## drop outcome column from data df
#data = data.drop('outcome', axis=1)
#
## Create an array to hold all match data
#label_array = labels.values
#data_array = data.values
#
## split our data 80 test 20 train  
#bot_X_train, bot_X_test, bot_y_train, bot_y_test = train_test_split(data_array, 
#                                                                   label_array, 
#                                                                test_size=0.20, 
#                                                             random_state=1337)

# Sets Y to ints since they are only 1, 0 or -1
#bot_y_train = bot_y_train.astype(np.int32)
#bot_y_test = bot_y_test.astype(np.int32)
#
## pull out 10% for validation
#bot_X_valid, bot_X_train = bot_X_train[:500], bot_X_train[500:]
#bot_y_valid, bot_y_train = bot_y_train[:500], bot_y_train[500:]

# creates a shuffler
# prevents the neural net from learning the order and not the data
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


#len(supply_array[0,:])
#len(actions_array[0,:])

# total input variables being recorded from the game
n_inputs = len(supply_array[0,:])
n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = 50
n_outputs = len(actions_array[0,:])

# set parameters for the NN
learning_rate = 0.0001

n_epochs = 10
batch_size = 25

# sets names for the tensorboard
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# 3 hidden layers
# 1 output layer
with tf.name_scope("dnn"):
    hidden_layer_1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, 
                                     name="hidden_layer_1")
    hidden_layer_2 = tf.layers.dense(hidden_layer_1, n_hidden2, 
                                     activation=tf.nn.relu, 
                                     name="hidden_layer_2")
    hidden_layer_3 = tf.layers.dense(hidden_layer_2, n_hidden3, 
                                     activation=tf.nn.relu, 
                                     name="hidden_layer_3")
    #logits = Dense(hidden_layer_3, n_outputs, name="outputs")
    logits = tf.layers.dense(hidden_layer_3, n_outputs, name="outputs")

    tf.summary.histogram('hidden_layer_1', hidden_layer_1)
    tf.summary.histogram('hidden_layer_2', hidden_layer_2)
    tf.summary.histogram('hidden_layer_3', hidden_layer_3)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
init = tf.global_variables_initializer()
merged_summaries = tf.summary.merge_all()

saver = tf.train.Saver()

means = supply_array_scaled_train_x.mean(axis=0, keepdims=True)
stds = supply_array_scaled_train_x.std(axis=0, keepdims=True) + 1e-10
X_val_scaled = (supply_array_scaled_test_x - means) / stds    

# async file saving objects
train_saver = tf.summary.FileWriter('./model/train', tf.get_default_graph())  
test_saver = tf.summary.FileWriter('./model/test')

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(supply_array_scaled_train_x, 
                                              supply_array_scaled_train_y, 
                                              batch_size):
            X_batch_scaled = (X_batch - means) / stds
            summaries, _ = sess.run([merged_summaries, training_op], 
                                    feed_dict={X: X_batch_scaled, y: y_batch})
        train_saver.add_summary(summaries, epoch)
        _, acc_batch = sess.run([merged_summaries, accuracy], 
                                feed_dict={X: X_batch_scaled, y: y_batch})
        train_summaries, acc_valid = sess.run([merged_summaries, accuracy], 
                                              feed_dict={X: X_val_scaled, 
                                                y: supply_array_scaled_test_y})
        test_saver.add_summary(train_summaries, epoch)
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", 
              acc_valid)

    train_saver.flush()