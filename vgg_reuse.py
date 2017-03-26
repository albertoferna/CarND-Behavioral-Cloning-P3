from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras import applications
from keras.regularizers import l2
from data_adaptation import *

data_folder = './data_ud/'

pre_model = applications.VGG16(include_top=False, input_shape=(85, 320, 3))
# Mark layers as not trainable
for layer in pre_model.layers:
    layer.trainable = False

model = Sequential()

model.add(pre_model)
model.add(Flatten())

model.add(Dense(1000, activation='relu', kernel_regularizer=l2(0.0001)))
#model.add(Dropout(0.5))

model.add(Dense(250, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.5))

model.add(Dense(1, kernel_initializer='glorot_normal'))

model.compile(loss='mse', optimizer='SGD')

# uncomment the following line to restart a training run from disk
# model.load_weights('vgg_reuse.h5')
saving = ModelCheckpoint('vgg_best.h5', save_best_only=True)
samples_per_training = 500

used_samples = read_data(data_folder, samples_per_training, 20)
X_train, y_train = get_training_data(used_samples, data_folder)
X_train = get_cropped_data(X_train)

model.fit(X_train, y_train, batch_size=128, validation_split=0.2, shuffle=True,
          epochs=8, callbacks=[saving])
model.save('vgg_reuse.h5')
