from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Cropping2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from data_adaptation import *

data_folder = './data_ud/'

used_samples = read_data(data_folder, 1000, 20)
X_train, y_train = get_training_data(used_samples, data_folder)

# Trying to use AlexNet as an inspiration
model = Sequential()
net_scaling = 6
# Image processing
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
# Layer 1
model.add(Convolution2D(96 // net_scaling, 11, 11, activation='relu', border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# Layer 2
model.add(Convolution2D(256 // net_scaling, 5, 5, activation='relu', border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# Layer 3
model.add(Convolution2D(512 // net_scaling, 3, 3, activation='relu', border_mode='valid'))

# Layer 4
model.add(Convolution2D(1024 // net_scaling, 3, 3, activation='relu', border_mode='valid'))

# Layer 5
model.add(Convolution2D(1024 //net_scaling, 3, 3, activation='relu', border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# Layer 6
model.add(Flatten())
model.add(Dense(3072 // net_scaling, activation='relu', init='glorot_normal'))
model.add(Dropout(0.5))

# Layer 7
model.add(Dense(4096 // net_scaling, activation='relu', init='glorot_normal'))
model.add(Dropout(0.5))

# Layer 8
model.add(Dense(1, init='glorot_normal'))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=128, validation_split=0.3, shuffle=True, nb_epoch=10)

model.save('model_AlexNet.h5')
