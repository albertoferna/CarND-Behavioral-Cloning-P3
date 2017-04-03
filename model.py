#!/usr/bin/env python
# Model based on Commaai
"""
Steering angle prediction model
"""
from data_adaptation import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2

data_folder = '../CarND-Data-P3/data_sect/'

used_samples = read_data(data_folder, samples_per_bin=100, bins=100)
X_train, y_train = get_training_data(used_samples, data_folder)

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/127.5 - 1., ))
model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding="same"))
model.add(ELU())
model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
model.add(ELU())
model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512, kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001)))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.load_weights('model_commaai.h5')
saving = ModelCheckpoint('commaai.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
model.fit(X_train, y_train, batch_size=32, validation_split=0.3, shuffle=True, epochs=8,
          callbacks=[saving, earlyStopping])
model.save('model_commaai.h5')
