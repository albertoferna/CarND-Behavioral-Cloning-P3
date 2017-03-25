import numpy as np
import random
import csv
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Cropping2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten, Dense, Dropout

lines = []
data_folder = './data_ud/'
with open(data_folder + 'driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

images_center = []
images_left = []
images_right = []
measurements = []

for line in random.sample(lines[1:], 1000):
    center_file = line[0].split('/')[-1]
    center_image = cv2.imread(data_folder+'IMG/'+center_file.strip())
    images_center.append(center_image)
    measurements.append(float(line[3]))
    flipped = cv2.flip(center_image.copy(), 1)
    images_center.append(flipped)
    measurements.append(-float(line[3]))

    left_file = line[1].split('/')[-1]
    left_image = cv2.imread(data_folder+'IMG/'+left_file.strip())
    images_left.append(left_image)
    measurements.append(float(line[3]) + 0.05)
    flipped = cv2.flip(left_image.copy(), 1)
    images_left.append(flipped)
    measurements.append((-float(line[3]) + 0.05))

    right_file = line[2].split('/')[-1]
    right_image = cv2.imread(data_folder+'IMG/'+right_file.strip())
    images_right.append(right_image)
    measurements.append(float(line[3]) - 0.05)
    flipped = cv2.flip(right_image.copy(), 1)
    images_right.append(flipped)
    measurements.append(-(float(line[3]) - 0.05))

images = images_left + images_right + images_center
X_train = np.array(images)
y_train = measurements

# Trying to use LeNet as an inspiration
model = Sequential()

# Image processing
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))

# Layer 1
model.add(Convolution2D(12, (5, 5), activation='relu', padding='valid', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Convolution2D(32, (5, 5), activation='relu', padding='valid', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Flatten())
model.add(Dense(120, activation='relu', kernel_initializer='TruncatedNormal'))
model.add(Dropout(0.5))

# Layer 4
model.add(Dense(84, activation='relu', kernel_initializer='TruncatedNormal'))
#model.add(Dropout(0.5))

# Layer 6
model.add(Dense(1, kernel_initializer='TruncatedNormal'))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=128, validation_split=0.3, shuffle=True, epochs=15)

model.save('model_LeNet.h5')
