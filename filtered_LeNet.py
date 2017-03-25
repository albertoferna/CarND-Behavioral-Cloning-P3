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

################# Pre-selection of samples to use #############
n = 100  # number of samples of angle zero to consider
b = 20   # number of bins to divide the angles in
samples_per_bin = 20  # Maximum samples per final bin
# Let's read all angles
angles_list = []
for line in lines[1:]:
    angles_list.append(float(line[3].strip()))
angles = np.array(angles_list)
# Cut too high or too low
chopped_index = (np.abs(angles) < 0.5)
chopped_lines = np.array(lines[1:])[chopped_index]
chopped_angles = angles[chopped_index]
# Remove most of zero steering angle samples
zero_angle_index = (chopped_angles == 0)
valid_angle_index= (chopped_angles != 0)
zero_angle_lines = chopped_lines[zero_angle_index]
valid_lines = chopped_lines[valid_angle_index]
valid_angles = chopped_angles[valid_angle_index]
# Select random samples from zero angle
idx = np.random.choice(len(zero_angle_lines) - 1, n, replace=False)
valid_lines = np.vstack((valid_lines, zero_angle_lines[idx]))
valid_angles = np.hstack((valid_angles, np.zeros(n)))
# Selecting final samples to use
hist = np.histogram(valid_angles, bins=b)
used_samples = []
bin_start = min(valid_angles)
for bin_end in hist[1][1:]:
    selectable = np.where(np.logical_and(valid_angles >= bin_start, valid_angles <= bin_end))
    samples_in_bin = len(valid_angles[selectable])
    if samples_in_bin > samples_per_bin:
        idx = np.random.choice(samples_in_bin, samples_per_bin, replace=False)
        select_from = valid_lines[selectable]
        used_samples.append(select_from[idx])
    else:
        used_samples.append((valid_lines[selectable]))
    bin_start = bin_end
used_samples = np.concatenate(used_samples)
####### Samples selected ########

for line in used_samples:
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
model.add(Convolution2D(24, (5, 5), activation='relu', padding='valid', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Convolution2D(64, (5, 5), activation='relu', padding='valid', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Flatten())
model.add(Dense(180, activation='relu', kernel_initializer='TruncatedNormal'))

# Layer 4
model.add(Dense(60, activation='relu', kernel_initializer='TruncatedNormal'))

# Layer 5
model.add(Dense(1, kernel_initializer='TruncatedNormal'))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=128, validation_split=0.3, shuffle=True, epochs=5)

model.save('filtered_LeNet.h5')
