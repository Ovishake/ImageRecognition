#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:50:00 2019

@author: ovishake
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from glob import glob
import os
xray_data = pd.read_csv('Data_Entry_2017.csv')
my_glob = glob('images/*.png')
full_img_paths = {os.path.basename(x): x for x in my_glob}
xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)

#unique findings
num_unique_labels = xray_data['Finding Labels'].nunique()
xray_data['Finding_strings']=xray_data['Finding Labels'].map(lambda x: x[:4])
print(xray_data['Finding_strings'].head(10))
#print(num_unique_labels)

#Ignoring the No Finding
dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

#creating multi hot tensor
for label in dummy_labels:
    xray_data[label] = xray_data['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
    
#Now creating the multi hot tensor in the dataframe to make it flow to the data generator
xray_data['target_vector'] = xray_data.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])

#print(xray_data.apply(lambda target: [target[dummy_labels].values], 1).head(10))
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(xray_data, test_size = 0.05, train_size=0.5)
print('{} train set'.format(train_set.shape))
print('{} test set'.format(test_set.shape))


from keras.preprocessing.image import ImageDataGenerator
data_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

#def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
#    base_dir = os.path.dirname(in_df[path_col].values[0])
#    print('## Ignore next message from keras')
#    df_gen = img_data_gen.flow_from_directory(base_dir,class_mode = 'sparse',**dflow_args)
#    df_gen.filenames = in_df[path_col].values
#    df_gen.classes = np.stack(in_df[y_col].values)
#    df_gen.samples = in_df.shape[0]
#    df_gen.n = in_df.shape[0]
#    df_gen._set_index_array()
#    df_gen.directory = '' # since we have the full path
#    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
#    return df_gen
image_size = (128, 128)
train_gen = data_gen.flow_from_dataframe(
        dataframe= train_set,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode= 'grayscale',
        class_mode="categorical",
        batch_size = 32)
valid_gen = data_gen.flow_from_dataframe(
        dataframe = test_set,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode='grayscale',
        class_mode="categorical",
        batch_size=64)

step_size_train= train_gen.n // train_gen.batch_size
step_size_valid = valid_gen.n // valid_gen.batch_size
test_x, test_y = next(data_gen.flow_from_dataframe(
        dataframe = test_set,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode='grayscale',
        class_mode="categorical",
        batch_size=64))

# image re-sizing target
#train_gen = flow_from_dataframe(data_gen, train_set, path_col = 'full_path', y_col = 'target_vector', target_size = image_size, color_mode = 'grayscale', batch_size = 32)
#valid_gen = flow_from_dataframe(data_gen, test_set, path_col = 'full_path', y_col = 'target_vector', target_size = image_size, color_mode = 'grayscale', batch_size = 128)

#Test Set
#test_X, test_Y = flow_from_dataframe(data_gen, test_set, path_col = 'full_path', y_col = 'target_vector', target_size = image_size, color_mode = 'grayscale',batch_size = 2048)d


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = [128,128,1]))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
          
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
          
model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(len(dummy_labels), activation = 'softmax'))

# compile model, run summary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='weights.best.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only = True)
callbacks_list = [checkpointer]

model.fit_generator(generator = train_gen, 
        steps_per_epoch = step_size_train, 
        epochs = 1, 
        callbacks = callbacks_list, 
        validation_data = (test_x, test_y),
        validation_steps=step_size_valid)

