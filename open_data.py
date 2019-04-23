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

# let's grab two classes
label2 = label[2:4]
y = list()
for i in range(len(labels)):
    for j in range(len(label2)):
        if labels[i] == label2[j]:
            y.append(j)
            print (j)

y = np.asarray(y)
print (np.array(np.unique(y, return_counts=True)).T)   

# Let's subset x data
train0 = train.file[train.label=='cat']
train1 = train.file[train.label=='dog']
df_train = pd.concat([train0,train1], axis=0)

files0 = train0.values.tolist()
files1 = train1.values.tolist()
files = files0 + files1

label0 = train.label[train.label=='cat']
label1 = train.label[train.label=='dog']
df_labels = pd.concat([label0,label1], axis=0)

df_train = pd.concat([df_train,df_labels], axis=1)

label0 = label0.values.tolist()
label1 = label1.values.tolist()
labels = label0 + label1

paths = list()
for i, file in enumerate(files):
    print(file)
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

<<<<<<< HEAD
x = np.expand_dims(x, axis=3)        
=======
x2 = np.expand_dims(x, axis=3)        
>>>>>>> c62feaadde2ac46e429e348b4fc114c6dc58126d

np.save('xdata.npy',x)
np.save('ydata.npy',y)

x = np.load('xdata.npy')
y = np.load('ydata.npy')
########################################################################
########################################################################
# Split data for training
from sklearn.metrics import classification_report
from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split

print ("Y", np.array(np.unique(y, return_counts=True)).T) 
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
x_imb, y_imb = make_imbalance(x, y, ratio={0: 1733, 1: 500}, random_state=0)
x = x.reshape(x.shape[0], 151, 161, 1)
x_imb = x_imb.reshape(x_imb.shape[0], 151, 161, 1)

# Split data train, test, validation (25 samples per class)
xtrain, xtest, ytrain, ytest = train_test_split(x_imb, y_imb, test_size=0.25, random_state=1)
xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.05, random_state=1)

# print counts
print ("Ytrain", np.array(np.unique(ytrain, return_counts=True)).T) 
print ("Ytest", np.array(np.unique(ytest, return_counts=True)).T)
print ("Yval", np.array(np.unique(yval, return_counts=True)).T)


########################################################################
########################################################################
# CNN Training
<<<<<<< HEAD

num_classes = np.max(ytrain)+1
=======
batch_size = 64
num_classes = len(y)
epochs = 12
>>>>>>> c62feaadde2ac46e429e348b4fc114c6dc58126d

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

input_shape = (151, 161, 1)

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

<<<<<<< HEAD
earlyStopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history = model.fit(xtrain, ytrain,
          batch_size=32,
          epochs=25,
=======
history = model.fit(xtrain, ytrain,
          batch_size=batch_size,
          epochs=epochs,
>>>>>>> c62feaadde2ac46e429e348b4fc114c6dc58126d
          verbose=1,
          validation_data=(xtest, ytest),
          callbacks=[earlyStopping, mcp_save])


y_pred = model.predict(xtest)
y_pred = np.argmax(y_pred, axis=-1)
target_names = ['class 0', 'class 1']
print(classification_report(ytest, y_pred, target_names=target_names))


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
fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(ax=ax, x="label", data=df_train)
print(df_train.label.unique())

plt.close()

def spectrogram2(file, label):
    eps=1e-10
    sample_rate, samples = wavfile.read(str(train_audio_path) + '/' + label + '/' + file)
    frequencies, times, spectrogram = signal.stft(samples, sample_rate, nperseg = sample_rate/50, noverlap = sample_rate/75)
    return np.log(np.abs(spectrogram).T+eps)


num_samples = 5
label = df_train.label.unique()
fig, axes = plt.subplots(len(label),num_samples, figsize = (16, len(label)*4))
for i,labels in enumerate(label):
    files = df_train[df_train.label==labels].file.sample(num_samples)
    axes[i][0].set_title(labels)
    for j, file in enumerate(files):
        specgram = spectrogram2(file, labels)
        print (specgram.shape)
        axes[i][j].axis('off')
        axes[i][j].matshow(specgram)