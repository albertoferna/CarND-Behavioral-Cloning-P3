import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Cropping2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from data_adaptation import *

data_folder = './data_ud/'

used_samples = read_data(data_folder, 1000, 20)
X_train, y_train = get_training_data(used_samples, data_folder)

# Trying to use LeNet as an inspiration
model = Sequential()

# Image processing
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# Layer 1
model.add(Convolution2D(12, (5, 5), activation='relu', padding='valid', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Convolution2D(32, (5, 5), activation='relu', padding='valid', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Flatten())
model.add(Dense(120, activation='relu', kernel_initializer='TruncatedNormal'))

# Layer 4
model.add(Dense(84, activation='relu', kernel_initializer='TruncatedNormal'))

# Layer 5
model.add(Dense(1, kernel_initializer='TruncatedNormal'))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=128, validation_split=0.3, shuffle=True, epochs=5)

model.save('filtered_LeNet.h5')
