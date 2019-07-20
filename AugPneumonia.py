#"""
#Created on Thu Jul 11 19:34:42 2019
#
#@author: ovishake
#"""
#import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, concatenate
from keras.layers import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import warnings
from keras.callbacks import ModelCheckpoint
import seaborn as sns

import pandas as pd

from glob import glob
import os

from custom_layers import Scale

import cv2

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import numpy as np

from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

bm = keras.applications.DenseNet121(
        weights = 'imagenet',
        include_top=False,
        input_shape=(224,224,3),
        pooling='avg')

x = bm.output
#x = keras.layers.Dropout(rate=0.3)(x)
pred= Dense(1,activation='sigmoid')(x)

model = keras.models.Model(inputs=bm.input, output=pred)

xray_data = pd.read_csv('train.csv')
#TAKING ONLY ONE DATA PER CLIENT based on PATIENT ID
xray_data = xray_data.drop_duplicates(subset='Patient ID')
fe = xray_data[xray_data['Finding Labels'] == "No Finding"]
fe = fe.head(1000)
xray_data = xray_data[xray_data['Finding Labels'] != "No Finding"]
xray_data = xray_data.append(fe)

my_glob = glob('images/*.png')
full_img_paths = {os.path.basename(x): x for x in my_glob}
xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)
print(len(xray_data['full_path']))
xray_data['Finding_strings']=xray_data['Finding Labels'].map(lambda x: '1' if 'Pneumonia' in x else '0')
#xray_data['Finding_strings']=pd.get_dummies(xray_data['Finding Labels'])

AP = xray_data[(xray_data['Finding Labels']=="Pneumonia") & (xray_data['View Position']=="AP")]
PA = xray_data[(xray_data['Finding Labels']=="Pneumonia") & (xray_data['View Position']=="PA")]

unaugmented_data = xray_data[(xray_data['Finding Labels']!="Pneumonia")]

#taking only unique patient ID for validation
xray_data_test = pd.read_csv('test.csv')
xray_data_test['full_path'] = xray_data_test['Image Index'].map(full_img_paths.get)
print(len(xray_data_test['full_path']))
xray_data_test['Finding_strings']=xray_data_test['Finding Labels'].map(lambda x: '1' if 'Pneumonia' in x else '0')
#xray_data_test['Finding_strings']=pd.get_dummies(xray_data['Finding Labels'])

xray_data_valid = pd.read_csv('valid.csv')
xray_data_valid['full_path'] = xray_data_valid['Image Index'].map(full_img_paths.get)
print(len(xray_data_valid['full_path']))
xray_data_valid['Finding_strings']=xray_data_valid['Finding Labels'].map(lambda x: '1' if 'Pneumonia' in x else '0')
#xray_data_valid['Finding_strings']=pd.get_dummies(xray_data['Finding Labels'])

test_labels = xray_data_valid['Finding_strings'].astype(float).values

from keras.preprocessing.image import ImageDataGenerator
data_gen = ImageDataGenerator(
        zoom_range=0.1,
        rotation_range=20,
        horizontal_flip=True)

data_gen2 = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

data_gen3 = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

vanilla_data_gen = ImageDataGenerator(
            rescale=1./255.0
            )

image_size = (224, 224)
#data from the df xray_data
train_gen = data_gen.flow_from_dataframe(
        dataframe= xray_data,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode= 'rgb',
        class_mode="binary",
        shuffle=True,
        batch_size = 32)
#data from the df xray_data_test
test_gen = data_gen2.flow_from_dataframe(
        dataframe = xray_data_test,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode='rgb',
        class_mode="binary",
        shuffle=True,
        batch_size=32)


test_x = data_gen3.flow_from_dataframe(
        dataframe = xray_data_valid,
        directory=None,
        x_col='full_path',
        y_col=None,
        target_size=image_size,
        color_mode='rgb',
        class_mode=None,
        shuffle=True,
        batch_size=32)

de1 = data_gen.flow_from_dataframe(
        dataframe= AP,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode= 'rgb',
        class_mode=None,
        shuffle=True,
        batch_size = 32)

de2 = data_gen.flow_from_dataframe(
        dataframe= AP,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode= 'rgb',
        class_mode=None,
        shuffle=True,
        batch_size = 32)

de3 = data_gen.flow_from_dataframe(
        dataframe= unaugmented_data,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode= 'rgb',
        class_mode=None,
        shuffle=True,
        batch_size = 32)

batch_images = []



for images in tqdm(AP['full_path'].values):
    img = cv2.imread(images,0)
    img = cv2.resize(img, dsize=(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    batch_images.append(img)
    img = cv2.flip(img, 0)
#    img = img/255.0
    batch_images.append(img)

for images in tqdm(PA['full_path'].values):
    img = cv2.imread(images,0)
    img = cv2.resize(img, dsize=(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    batch_images.append(img)
    img = cv2.flip(img, 0)
#    img = img/255.0
    batch_images.append(img)

print('Total Synthetic Data Generation', len(batch_images))

augmented_labels = np.ones(len(batch_images), dtype=np.int32).astype(str)


for images in tqdm(unaugmented_data['full_path'].values):
    img = cv2.imread(images,0)
    img = cv2.resize(img, dsize=(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#    img = img/255.0
    batch_images.append(img)
    
print('Total training examples', len(batch_images))
    
batch_images = np.array(batch_images, dtype=np.float32)

unaug_labels = unaugmented_data['Finding_strings'].values

train_labels = np.append(augmented_labels, unaug_labels, axis=0)

optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.95, amsgrad=True)
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
filepath = "Pneumonia.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', period=3, save_best_only=True)

callbacks_list = [checkpoint]

step_size_train= train_gen.n // train_gen.batch_size
step_size_valid = test_gen.n // test_gen.batch_size

grand_augmented_data = vanilla_data_gen.flow(
        batch_images, train_labels, shuffle=True, batch_size= 32)
steps_per_epoc = grand_augmented_data.n / 32
history = model.fit_generator(generator = grand_augmented_data, 
        steps_per_epoch = steps_per_epoc, 
        epochs = 100, 
        callbacks = callbacks_list, 
        validation_data = test_gen,
        validation_steps=step_size_valid)

model_yaml = model.to_yaml()
out_file = open("Pneumonia.yaml","w")

out_file.write(model_yaml)
out_file.close()

model.save_weights('Pneumonia_1.h5')

training_loss = history.history['loss']
test_loss = history.history['val_loss']

accuracy = history.history['acc']

testing_accuracy = history.history['val_acc']
#test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss) + 1)

sns.set()
g = sns.lineplot(x= training_loss, y=epoch_count)
g.figure.savefig('../charts and graphs/trainingloss for Pneumonia.png')

h = sns.lineplot(x= test_loss, y=epoch_count)
h.figure.save('../charts and graphs/testingloss for Pneumonia.png')

i = sns.lineplot(x= accuracy, y=epoch_count)
i.figure.save('../charts and graphs/accuracy for Pneumonia.png')

j = sns.lineplot(x= testing_accuracy, y=epoch_count)
j.figure.save('../charts and graphs/validation accuracy for Pneumonia.png')

