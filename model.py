#!/usr/bin/env python
# Model based on Commaai
"""
Steering angle prediction model
"""
from data_adaptation import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

data_folder = '../CarND-Data-P3/data_final/'

used_samples = read_data(data_folder, samples_per_bin=500, bins=200)
X_train, y_train = get_training_data(used_samples, data_folder)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(h, w, 3)))
model.add(Convolution2D(3, (1, 1), strides=(1, 1), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(ELU())
model.add(Convolution2D(16, (5, 5), strides=(4, 4), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(ELU())
model.add(Convolution2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(ELU())
model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal'))

model.add(Flatten())
model.add(Dropout(.2))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
model.add(Dropout(.5))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse", lr=0.001)
model.load_weights('model_commaai.h5')
saving = ModelCheckpoint('commaai.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
model.fit(X_train, y_train, batch_size=32, validation_split=0.2, epochs=20, shuffle=True,
          callbacks=[saving, earlyStopping])
model.save('model_commaai.h5')
