from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Cropping2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l2
from data_adaptation import *

data_folder = './data_mix/'

used_samples = read_data(data_folder, 2000, 20)
X_train, y_train = get_training_data(used_samples, data_folder)

# Trying to use LeNet as an inspiration
model = Sequential()

# Image processing
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# Layer 1
model.add(Convolution2D(6, (5, 5), activation='relu', padding='valid', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Convolution2D(16, (5, 5), activation='relu', padding='valid',
                        kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Flatten())
model.add(Dense(120, activation='relu', kernel_initializer='TruncatedNormal', kernel_regularizer=l2(0.0001)))
#model.add(Dropout(0.5))

# Layer 4
model.add(Dense(84, activation='relu', kernel_initializer='TruncatedNormal', kernel_regularizer=l2(0.0001)))

# Layer 5
model.add(Dense(1, kernel_initializer='TruncatedNormal'))

model.compile(loss='mse', optimizer='adam')

# uncomment the following line to restart a training run from disk
model.load_weights('model_LeNet.h5')
saving = ModelCheckpoint('LeNet_best.h5', save_best_only=True)
model.fit(X_train, y_train, batch_size=128, validation_split=0.2, shuffle=True, epochs=4, callbacks=[saving])
model.save('model_LeNet.h5')

# Overfitting check

"""im = X_train[0].copy()
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(im,'Steering: ' + str(y_train[0]), (70, 20), font, 0.5, (255, 255, 255), 1)
cv2.imwrite('./examples/overfit.jpg', im)"""

