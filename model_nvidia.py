#!/usr/bin/env python
# Model based on Nvidia's
"""
Steering angle prediction model
"""
from data_adaptation import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2

model = Sequential()
model.add(Cropping2D(cropping=((55, 35), (0, 0)), input_shape=(160, 320, 3)))
# Normalize the input without changing shape
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(50, 320, 3), output_shape=(70, 320, 3)))
# Convolution layers
model.add(Convolution2D(24, (5, 5), padding='valid', kernel_initializer='glorot_uniform', strides=(2, 2)))
model.add(ELU())
model.add(Convolution2D(36, (5, 5), padding='valid', kernel_initializer='glorot_uniform', strides=(2, 2)))
model.add(ELU())
model.add(Convolution2D(48, (5, 5), padding='valid', kernel_initializer='glorot_uniform', strides=(2, 2)))
model.add(ELU())
model.add(Convolution2D(64, (3, 3), padding='valid', kernel_initializer='glorot_uniform', strides=(1, 1)))
model.add(ELU())
model.add(Convolution2D(64, (3, 3), padding='valid', kernel_initializer='glorot_uniform', strides=(1, 1)))

# Fully connected classifier
model.add(Flatten())
model.add(Dropout(0.25))
model.add(ELU())

model.add(Dense(1164, kernel_initializer='glorot_uniform'))
model.add(Dropout(0.25))
model.add(ELU())

model.add(Dense(100, kernel_initializer='glorot_uniform'))
model.add(Dropout(0.25))
model.add(ELU())

model.add(Dense(50, kernel_initializer='glorot_uniform'))
model.add(Dropout(0.25))
model.add(ELU())

model.add(Dense(10, kernel_initializer='glorot_uniform'))
model.add(Dropout(0.25))
model.add(ELU())

model.add(Dense(1, kernel_initializer='glorot_uniform', name='output'))

# Display the model summary
#model.summary()
# Compile it
model.compile(optimizer="adam", loss="mse", lr=0.0001)
saving = ModelCheckpoint('nvidia.{epoch:02d}-{val_loss:.3f}.h5', save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

data_folder = '../CarND-Data-P3/data_all/'

# Data read is picked randomly with a certain distribution. So this can run several times
used_samples = read_data(data_folder, samples_per_bin=80, bins=250)
X_train, y_train = get_training_data(used_samples, data_folder)
model.fit(X_train, y_train, batch_size=128, validation_split=0.2, shuffle=True, epochs=50,
          callbacks=[saving, earlyStopping])

model.save('model_nvidia.h5')
