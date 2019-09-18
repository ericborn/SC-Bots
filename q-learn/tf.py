import tensorflow as tf
import keras
from keras.models import Sequential
#from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
import os
import random

## Tests if tf is up and ready
#tf.test.is_built_with_cuda()
#
## Tests if GPU is ready
#tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# path to the training data
path = 'C:/Users/TomBrody/Desktop/Projects/SC-Bots/q-learn/training/training_data/'

# list of all files
files = os.listdir(path)

# column list for all match data
cols = ['supply_cap', 'supply_army', 'supply_workers', 'pylons', 
        'assimilators', 'gateways', 'stalkers', 'voidrays', 'nexus', 
        'd-structures', 'd-units', 'outcome']

# define empty dataframe using columns previously defined
full_df = pd.DataFrame(columns=cols)

# for loop that evaluates the numpy files in index 11 to look for match outcome
# 1 means win, 0 means tie, -1 means loss
for file in range(0, len(files)):
    full_df.loc[file] = np.load(path + files[file])

# creates labels and data sets
labels = full_df['outcome']
data = full_df

# drop outcome column from data df
data = data.drop('outcome', axis=1)

# Create an array to hold all match data
label_array = labels.values
data_array = data.values

# shape data into batches
samples = list()
length = 200

# iterate over all samples 200 at a time
for i in range(0,len(data_array),length):
	# select from i to i + 200
	sample = data_array[i:i+length]
	samples.append(sample)
print(len(samples))

# Create a new array based upon the sample size of 200
data = np.array(samples)

# The LSTM needs data with the format of [samples, time steps and features]
# toal samples, size of each sample, and total features
# 25, 200, 11
data = data.reshape(len(samples), length, 11)
print(data.shape)

# Create keras model
model = Sequential()

# sendex
# add layers
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 11),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/Stage1")

model.fit(data, label_array, epochs=10, batch_size=200)

test_size = 100

# feed both objects into keras

epics = 10
batch = 100

model.fit(x_train, y_train,
          batch_size=batch,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=1, callbacks=[tensorboard]) 
    
    

files[0:10]

test_size = 100


for file in all_files[current:current+increment]:
    full_path = os.path.join(train_data_dir, file)
    data = np.load(full_path)
    data = list(data)
    for d in data:
        choice = np.argmax(d[0])
        if choice == 0:
            no_attacks.append([d[0], d[1]])
        elif choice == 1:
            attack_closest_to_nexus.append([d[0], d[1]])
        elif choice == 2:
            attack_enemy_structures.append([d[0], d[1]])
        elif choice == 3:
            attack_enemy_start.append([d[0], d[1]])



x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3)






path = tf.keras.utils.get_file('mnist.npz', DATA_URL)













