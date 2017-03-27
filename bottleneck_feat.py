from data_adaptation import *
from keras import applications
import numpy as np
import glob

data_folder = './data_mix/'
dump_folder = './data_vgg/'

# Read latest file in directory
files = glob.glob(dump_folder + 'bottleneck_*')
numbers = [int(x.split('/')[-1].split('.')[0][-3:]) for x in files]
if numbers:
    latest = max(numbers) + 1
else:
    latest = 0

lines = []
with open(data_folder + 'driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

X_train, y_train = get_training_data(lines[1:5000], data_folder)
X_train = get_cropped_data(X_train)
# remove zero angle samples
idx = y_train != 0
X_train = X_train[idx]
y_train = y_train[idx]

pre_model = applications.VGG16(include_top=False, input_shape=(85, 320, 3))
# Generate bottleneck features and save them
print('Dumping', len(X_train), 'samples after feeding through convolution')
samples_batch = 500
for batch in range(len(X_train) // samples_batch + 1):
    start = batch * samples_batch
    print('dumping batch:', batch, 'stating at', start)
    end = min((batch + 1) * samples_batch, len(X_train))
    train_data = pre_model.predict(X_train[start:end, :, :, :])
    train_data.dump(dump_folder + 'bottleneck_feat' + "{:03d}".format(batch + latest) + '.p')
    train_angle = np.array(y_train[start:end])
    train_angle.dump(dump_folder + 'angles' + "{:03d}".format(batch + latest) + '.p')
    print(end, 'samples processed')
