# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:57:29 2019

@author: zlangford
"""

import os
from pathlib import Path
from subprocess import check_output

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns


########################################################################
########################################################################
# Prepare Data 
folders = os.listdir("train/audio")
print(folders)


train_audio_path = 'train/audio'
train_labels = os.listdir(train_audio_path)
print ('Number of labels:', len(train_labels))

wavs = []
labels = []
for label in train_labels:
    if label == '_background_noise_':
        continue
    files = os.listdir(train_audio_path + '/' + label)
    for f in files:
        if not f.endswith('wav'):
            continue
        wavs.append(f)
        labels.append(label)

train = pd.DataFrame({'file':wavs,'label':labels})
train.info()
label = train.label.unique()

y = list()
for i in range(len(labels)):
    for j in range(len(label)):
        if labels[i] == label[j]:
            y.append(j)
            print (j)

y = np.asarray(y)
print (np.array(np.unique(y, return_counts=True)).T)   

 
'''
bed
bird
cat
dog
down
eight
five
four
go
happy
house
left
marvin
nine
no
off
on
one
right
seven
sheila
six
stop
three
tree
two
up
wow
yes
zero
'''

files = train.file
files = files.values.tolist()

paths = list()
for i, file in enumerate(files):
    print(i)
    label = labels[i]
    file = files[i]
    path = str(train_audio_path) + '/' + label + '/' + file
    paths.append(path)

from PIL import Image
def spectrogram(path):
    eps=1e-10
    sample_rate, samples = wavfile.read(path)
    frequencies, times, spectrogram = signal.stft(samples, sample_rate, nperseg = sample_rate/50, noverlap = sample_rate/75)
    img = np.log(np.abs(spectrogram).T+eps)
    im = Image.fromarray(img)
    img = im.resize((161,151), Image.ANTIALIAS)
    img = np.asarray(img)
    return img


x = np.empty((0, 151, 161))  
x.shape     
for i, file in enumerate(paths):
    print (i)
    img = spectrogram(file)
    x = np.append(x, [img], axis=0)

x2 = np.expand_dims(x, axis=3)        


########################################################################
########################################################################
# Split data for training
from sklearn.metrics import classification_report
from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split

# Split data train, test, validation
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=1)
xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=1)

# print counts
print ("Ytrain", np.array(np.unique(ytrain, return_counts=True)).T) 
print ("Ytest", np.array(np.unique(ytrain, return_counts=True)).T)
   
#xtrain, ytrain = make_imbalance(xtrain, ytrain, ratio={0: 1709813, 1: 104796, 2: 0}, random_state=0)


########################################################################
########################################################################
# CNN Training
batch_size = 64
num_classes = len(y)
epochs = 12

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

input_shape = (161, 151, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(xtrain, ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xtest, ytest))

score = model.evaluate(xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

########################################################################
########################################################################
# Plots
fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(ax=ax, x="label", data=train)
print(train.label.unique())

plt.close()

def spectrogram2(file, label):
    eps=1e-10
    sample_rate, samples = wavfile.read(str(train_audio_path) + '/' + label + '/' + file)
    frequencies, times, spectrogram = signal.stft(samples, sample_rate, nperseg = sample_rate/50, noverlap = sample_rate/75)
    return np.log(np.abs(spectrogram).T+eps)


num_samples = 5
label = train.label.unique()
fig, axes = plt.subplots(len(labels),num_samples, figsize = (16, len(labels)*4))
for i,label in enumerate(labels):
    files = train[train.label==label].file.sample(num_samples)
    axes[i][0].set_title(label)
    for j, file in enumerate(files):
        specgram = spectrogram2(file, label)
        axes[i][j].axis('off')
        axes[i][j].matshow(specgram)