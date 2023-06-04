# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#这些import会大大影响运行速度
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

train=pd.read_csv('Reference/csv/genere_features.csv')


def encode(train):
    label_encoder = LabelEncoder().fit(train.genre)
    labels = label_encoder.transform(train.genre)    
    classes = list(label_encoder.classes_)
    train = train.drop(['genre','file name'],axis=1)
    return train, labels, classes

train, labels, classes = encode(train)
scaler = StandardScaler().fit(train.values)
scaled_train = scaler.transform(train.values)

sss=StratifiedShuffleSplit(test_size=0.1,random_state=23)
for train_index, valid_index in sss.split(scaled_train,labels):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]

nb_class=len(classes)
nb_features=24

labels = np_utils.to_categorical(labels, nb_class)
y_train = np_utils.to_categorical(y_train, nb_class)
y_valid = np_utils.to_categorical(y_valid, nb_class)

if(os.path.exists('1dconv_MC.h5')):
    model=load_model('1dconv_MC.h5')
else:
    model = Sequential()
    model.add(Dense(48,input_shape=(nb_features,),activation='relu'))
    model.add(Dense(24,activation='relu'))
#    model.add(Dense(19,activation='relu'))
#    model.add(Dense(17,activation='relu'))
    model.add(Dense(nb_class,activation='softmax'))
    sgd = SGD(lr=0.001,nesterov=True,decay=1e-6,momentum=0.9)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

nb_epoch = 1000
model.fit(X_train,y_train,epochs=nb_epoch, validation_data=(X_valid, y_valid), batch_size=300)

model.save('1dconv_MC.h5')