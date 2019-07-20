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
        include_top=False,
        input_shape=(224,224,3),
        pooling='max')

x = bm.output
x = keras.layers.Dropout(rate=0.3)(x)
pred= Dense(1,activation='sigmoid')(x)

model = keras.models.Model(inputs=bm.input, output=pred)

model.load_weights('Pneumonia.h5')

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
        rescale=1./255,
        zoom_range=0.1,
        rotation_range=20,
        horizontal_flip=True)

data_gen2 = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.1,
        rotation_range=20,
        horizontal_flip=True)

data_gen3 = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 20,
        zoom_range = 0.1,
        horizontal_flip=True)

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
        batch_size = 16)
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
        batch_size=16)


test_x = data_gen3.flow_from_dataframe(
        dataframe = xray_data_valid,
        directory=None,
        x_col='full_path',
        y_col=None,
        target_size=image_size,
        color_mode='rgb',
        class_mode=None,
        shuffle=True,
        batch_size=16)

model.trainable = False
optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.9, amsgrad=True)
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

batch_img = []
cnt = 1
for i in tqdm(xray_data_valid['full_path'].values):
    try:
        img = cv2.imread(i,0)
        img = cv2.resize(img, dsize=(224,224))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img/255.0
        batch_img.append(img)
    except Exception as e:
        print(str(e))
    cnt += 1

batch_img = np.array(batch_img, dtype = np.float32)
    
y_pred = model.predict(batch_img, batch_size=16)

print(y_pred.shape)

print(test_labels.shape)

auc_keras = roc_auc_score(test_labels, y_pred)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, y_pred)

print(auc_keras)

sns.set()

g = sns.lineplot(y = tpr_keras, x = fpr_keras)

g.figure.savefig('../charts and graphs/Pneumonia001.png')