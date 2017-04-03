import numpy as np
import csv
import cv2


def toYUV(img):
    """ Needed because the image sent by the simulator is in RGB, not BGR"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def read_data(data_folder, samples_per_bin=100, bins=100):
    """ Helper function to select what data to use before training the networks
    It takes a data folder, max of samples per bin to to select and
    the number of bins used to balance how distributed angles are"""

    read_lines = []

    with open(data_folder + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            read_lines.append(line)
    lines = np.array(read_lines[1:])


    # Let's read all angles
    angles_list = []
    for line in lines:
        angles_list.append(float(line[3].strip()))
    angles = np.array(angles_list)

    # Selecting final samples to use
    hist = np.histogram(angles, bins=bins)
    used_samples = []
    # make use extremes are included
    hist[1][-1] += 0.0000001
    bin_start = min(angles)
    # Here we try to balance the samples by setting a maximum per bin
    for bin_end in hist[1][1:]:
        selectable = np.where(np.logical_and(angles >= bin_start, angles < bin_end))
        samples_in_bin = len(angles[selectable])
        if samples_in_bin > samples_per_bin:
            idx = np.random.choice(samples_in_bin, samples_per_bin, replace=False)
            select_from = lines[selectable]
            used_samples.append(select_from[idx])
        else:
            used_samples.append((lines[selectable]))
        bin_start = bin_end
    used_samples = np.concatenate(used_samples)

    return used_samples


def get_training_data(lines, data_folder):
    """This function takes a list or np array with a line from the log as each element, and the folder for that log
    It returns the full array of images ready for learning, and their value"""
    images_center = []
    images_left = []
    images_right = []
    measurements = []
    for line in lines:
        center_file = line[0].split('/')[-1]
        center_image = cv2.imread(data_folder + 'IMG/' + center_file.strip())
        images_center.append(toYUV(center_image))
        measurements.append(float(line[3]))
        flipped = cv2.flip(center_image.copy(), 1)
        images_center.append(toYUV(flipped))
        measurements.append(-float(line[3]))

        left_file = line[1].split('/')[-1]
        left_image = cv2.imread(data_folder + 'IMG/' + left_file.strip())
        images_left.append(toYUV(left_image))
        measurements.append(float(line[3]) + 0.25)
        flipped = cv2.flip(left_image.copy(), 1)
        images_left.append(toYUV(flipped))
        measurements.append(-(float(line[3]) + 0.25))

        right_file = line[2].split('/')[-1]
        right_image = cv2.imread(data_folder + 'IMG/' + right_file.strip())
        images_right.append(toYUV(right_image))
        measurements.append(float(line[3]) - 0.25)
        flipped = cv2.flip(right_image.copy(), 1)
        images_right.append(toYUV(flipped))
        measurements.append(-(float(line[3]) - 0.25))

    images = images_center + images_left + images_right
    return np.array(images), np.array(measurements)


def generate_train_data(train_lines, data_folder, batch_size):
    samples = batch_size // 6
    while 1:
        # shuffle samples to use
        np.random.shuffle(train_lines)
        # loop to use most of the data
        for i in range(0, len(train_lines) // samples):
            # select lines in this batch
            lines = train_lines[samples*i:samples*(i+1)]
            # get those sample lines and augment them
            X_train_batch, y_train_batch = get_training_data(lines, data_folder)
            yield X_train_batch, y_train_batch


def generate_val_data(val_lines, data_folder):
    while 1:
        # shuffle samples to use
        np.random.shuffle(val_lines)
        for line in val_lines[0:2]:
            center_file = line[0].split('/')[-1]
            center_image = cv2.imread(data_folder + 'IMG/' + center_file.strip())
            angle = np.array([[float(line[3])]])
            yield center_image[None, :, :, :], angle

