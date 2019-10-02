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
from keras.utils import to_categorical
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

# sequence length before making a prediction
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
EPOCHS = 10
BATCH_SIZE = 64  
NAME = f"{SEQ_LEN}-SEQ-RNN-Model{int(time.time())}"

# path to the training data
#path = 'C:/Users/TomBrody/Desktop/Projects/SC-Bots/q-learn/training/training_data/'
#path = 'C:/Users/TomBrody/Desktop/test/data/'
#send_path = 'C:/Users/TomBrody/Desktop/send/'
#send_files = os.listdir(send_path)
#full_array = np.load(send_path + send_files[0], allow_pickle=True)

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

# setup x and y
# x will be the supply_data inputs
# y will be the action that was taken

# Setup scalers for supply_array
scaler = StandardScaler()
scaler.fit(supply_array)
supply_array_scaled = scaler.transform(supply_array)


#x_train_shape[0][0][0]

# 80% train 20% test
(supply_array_scaled_train_x, supply_array_scaled_test_x, 
 supply_array_scaled_train_y, supply_array_scaled_test_y) = (
        train_test_split(supply_array_scaled, actions_array, test_size = 0.20, 
                         random_state=1337))


x_train_shape = supply_array_scaled_train_x.reshape(
        (1, supply_array_scaled_train_x.shape[0],
         supply_array_scaled_train_x.shape[1]))

y_train_shape = supply_array_scaled_train_y.reshape(
        (1, supply_array_scaled_train_y.shape[0],
         supply_array_scaled_train_y.shape[1]))

action_list = []

for i in range(0, len(y_train_shape)):
    for k in range(0, 9):
        if y_train_shape[i][k] == 1:
            action_list.append(k + 1)


#y_train_shape = to_categorical(supply_array_scaled_train_y, num_classes=9, 
#                               dtype='float32')

# setup the shapes for the input data
x_train_shape = supply_array_scaled_train_x.reshape(supply_array_scaled_train_x.shape[0], 
                                              supply_array_scaled_train_x.shape[1],1)

y_train_shape = supply_array_scaled_train_y.reshape(supply_array_scaled_train_y.shape[0], 
                                              supply_array_scaled_train_y.shape[1])

action_array = np.asarray(action_list)

action_array = action_array.reshape((341, 1))

#action_array = to_categorical(action_array)

#x_train_shape.shape
#y_train_shape.shape

y_train_shape[0][0]

# number of columns in our training data
#n_cols = supply_array_scaled_train_x.shape[1]
#n_rows = supply_array_scaled_train_x.shape[0]


# number of columns in our training data
#n_cols = x_train_shape.shape[1]
#n_rows = x_train_shape.shape[0]

#supply_array_scaled_train_y.shape

# batch size, time steps, number of units in one sequence
# 1, shape[0], shape[1]

# batch size can be omitted or added later, leaving just time steps and units
# input_shape(shape[0], shape[1])
# (341, 15)

# input_shape takes 2 inputs, rows, columns
# batch_input_shape takes 3 inputs, batch size, rows, columns

# overall shape shape = number of samples (games), 
# time steps (observations within the game), number of features (supply_data)
# 1, 341, 15

# input shape only takes timesteps and variables
# 341, 15

# start building the NN model
# Sequential allows the model to be build one layer at a time
# with each subsequent layer being added to the first
model = Sequential()
model.add(LSTM(9, input_shape=x_train_shape.shape[1:],
               return_sequences=True, activation='sigmoid'))
#model.add(LSTM(9, input_shape=(341,15)))
#model.add(Flatten())
#model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['categorical_accuracy'])
print(model.summary())
history = model.fit(x_train_shape, action_array, epochs=2, batch_size=1)



model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())

model.add(CuDNNLSTM(9, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(9))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# unique file name that will include the epoch 
# and the validation acc for that epoch
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  
# only saves the best ones
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, 
                             monitor='val_acc', verbose=1, save_best_only=True,
                             mode='max')) 

model.summary()

# Train model
history = model.fit(
    supply_array_scaled_train_x, supply_array_scaled_train_y,
    epochs=100,
    batch_size=64,
    validation_data=(supply_array_scaled_test_x, supply_array_scaled_test_y),
    #callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(supply_array_scaled_test_x,
                       supply_array_scaled_test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))