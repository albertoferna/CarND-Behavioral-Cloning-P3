from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from data_adaptation import *
import glob

data_folder = './data_vgg/'
x_data_files = glob.glob(data_folder + 'bottleneck*')
y_data_files = glob.glob(data_folder + 'angles*')
X = []
Y = []
for f_x, f_y in zip(x_data_files, y_data_files):
    X.append(np.load(f_x))
    Y.append(np.load(f_y))

X_train = np.concatenate(X)
y_train = np.concatenate(Y)

model = Sequential()

model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1000, activation='elu', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001)))

model.add(Dense(250, activation='elu', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001)))

model.add(Dense(1, kernel_initializer='glorot_normal'))

model.compile(loss='mse', optimizer='adam', lr=0.0001)

# uncomment the following line to restart a training run from disk
#model.load_weights('vgg_reuse.h5')
saving = ModelCheckpoint('vgg_best.h5', save_best_only=True)

model.fit(X_train, y_train, batch_size=256, validation_split=0.3, shuffle=True,
          epochs=30, callbacks=[saving])
model.save('vgg_reuse.h5')
