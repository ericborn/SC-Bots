# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:30:56 2019

@author: Eric Born
"""

import keras  # Keras 2.1.2 and TF-GPU 1.8.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3),
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
model.add(Dense(4, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/STAGE1")

train_data_dir = 'C:/Users/TomBrody/Desktop/send'


def check_data():
    choices = {"no_attacks": no_attacks,
               "attack_closest_to_nexus": attack_closest_to_nexus,
               "attack_enemy_structures": attack_enemy_structures,
               "attack_enemy_start": attack_enemy_start}

    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:",total_data)
    return lengths


x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3)
y_train = np.array([i[0] for i in train_data[:-test_size]])


# if you want to load in a previously trained model
# that you want to further train:
# keras.models.load_model(filepath)
hm_epochs = 10


# loop for total number of epochs
for i in range(hm_epochs):
    # sets iterator variables
    current = 0
    increment = 200
    
    #flag to stop the while loop
    not_maximum = True
    
    # creates a list of all files
    all_files = os.listdir(train_data_dir)
    
    #total number of training data files
    maximum = len(all_files)
    
    #shuffles file order
    random.shuffle(all_files)

    # loop to cycle through all files
    # while not_maximum:
    for i in range(0, 1):
        # output what range its working on
        print("WORKING ON {}:{}".format(current, current+increment))
        
        # empty lists to store ????
        no_attacks = []
        attack_closest_to_nexus = []
        attack_enemy_structures = []
        attack_enemy_start = []

        for file in all_files[current:current+increment]:
            full_path = os.path.join(train_data_dir, file)
            data = np.load(full_path, allow_pickle=True)
            data = list(data)
            print('im data', data)
            
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

        lengths = check_data()
        lowest_data = min(lengths)

        random.shuffle(no_attacks)
        random.shuffle(attack_closest_to_nexus)
        random.shuffle(attack_enemy_structures)
        random.shuffle(attack_enemy_start)

        no_attacks = no_attacks[:lowest_data]
        attack_closest_to_nexus = attack_closest_to_nexus[:lowest_data]
        attack_enemy_structures = attack_enemy_structures[:lowest_data]
        attack_enemy_start = attack_enemy_start[:lowest_data]

        check_data()

        train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start
        
        print('im train_data\n',train_data)
        
        random.shuffle(train_data)
        #print(len(train_data))

        test_size = 100
        batch_size = 128

        x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3)
        y_train = np.array([i[0] for i in train_data[:-test_size]])

        x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
        y_test = np.array([i[0] for i in train_data[-test_size:]])
        
#        model.fit(x_train, y_train,
#                  batch_size=batch_size,
#                  validation_data=(x_test, y_test),
#                  shuffle=True,
#                  verbose=1, callbacks=[tensorboard])
#
#        model.save("BasicCNN-{}-epochs-{}-LR-STAGE1".format(hm_epochs, learning_rate))
#        current += increment
#        if current > maximum:
#            not_maximum = False