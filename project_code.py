import glob
import tensorflow as tf
from PIL import Image
import numpy as np
from numpy import asarray
import os

#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def isFake(name):
  return int(name[-5])

files = glob.glob('C:\Users\desus\Desktop\Alcovex\images_from_video_big\*')
i = 0
for f in files:
  if i<10:
    print(f)
  else:
    break
  i+=1

first = True
i =0
for f in files:
    img = np.array(Image.open(f))
    if i<82000:
        if img.shape[0]==160:
            if first:
                X = [img]
                y = [isFake(f)]
                first= False
            else:
                X = np.concatenate((X,[img]))
                y.append(isFake(f))
            if i%1000==0:
                print(i//1000,f)
            i+=1
    else:
        break

y=to_categorical(y,2)

def build_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (160, 160, 3)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (160, 160, 3)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    return model

X_train, X_val, Y_train, Y_val = train_test_split(X[:1000], y[:1000], test_size = 0.2, random_state=5)

x_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(Y_train)
x_val = tf.convert_to_tensor(X_val)
y_val = tf.convert_to_tensor(Y_val)

print(tf.__version__)

model = build_model()

epochs = 10
batch_size = 64
init_lr = 1e-1
optimizer = Adam(lr = init_lr, decay = init_lr/epochs)
early_stopping = EarlyStopping(monitor = 'val_acc',
                              min_delta = 0,
                              patience = 2,
                              verbose = 0,
                              mode = 'auto')

model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

hist = model.fit(X_train,
                 Y_train,
                 batch_size = batch_size,
                 epochs = epochs,
                validation_data = (X_val, Y_val),
                callbacks = [early_stopping])