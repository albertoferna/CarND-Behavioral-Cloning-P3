import numpy as np
import csv
import cv2


def toRGB(img):
    """ Needed because the image sent by the simulator is in RGB, not BGR"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_data(data_folder, samples, bins):
    """ Helper function to select what data to use before training the networks
    It takes a data folder, an approximate number of total samples to select and
    the number of bins used to balance how distributed angles are"""
    lines = []
    n = int(2 * samples / bins)  # number of samples of angle zero to consider
    samples_per_bin = samples // bins

    with open(data_folder + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    # Let's read all angles
    angles_list = []
    for line in lines[1:]:
        angles_list.append(float(line[3].strip()))
    angles = np.array(angles_list)
    # Cut too high or too low
    chopped_index = (np.abs(angles) < 1.0)
    chopped_lines = np.array(lines[1:])[chopped_index]
    chopped_angles = angles[chopped_index]
    # Remove most of zero steering angle samples
    zero_angle_index = (chopped_angles == 0)
    valid_angle_index = (chopped_angles != 0)
    zero_angle_lines = chopped_lines[zero_angle_index]
    valid_lines = chopped_lines[valid_angle_index]
    valid_angles = chopped_angles[valid_angle_index]
    # Select random samples from zero angle
    if n > len(zero_angle_lines):
        # select all samples
        n = len(zero_angle_lines) - 1
    idx = np.random.choice(len(zero_angle_lines) - 1, n, replace=False)
    valid_lines = np.vstack((valid_lines, zero_angle_lines[idx]))
    valid_angles = np.hstack((valid_angles, np.zeros(n)))
    # Selecting final samples to use
    hist = np.histogram(valid_angles, bins=bins)
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
        images_center.append(toRGB(center_image))
        measurements.append(float(line[3]))
        flipped = cv2.flip(center_image.copy(), 1)
        images_center.append(toRGB(flipped))
        measurements.append(-float(line[3]))

        left_file = line[1].split('/')[-1]
        left_image = cv2.imread(data_folder + 'IMG/' + left_file.strip())
        images_left.append(toRGB(left_image))
        measurements.append(float(line[3]) + 0.1)
        flipped = cv2.flip(left_image.copy(), 1)
        images_left.append(toRGB(flipped))
        measurements.append(-(float(line[3]) + 0.1))

        right_file = line[2].split('/')[-1]
        right_image = cv2.imread(data_folder + 'IMG/' + right_file.strip())
        images_right.append(toRGB(right_image))
        measurements.append(float(line[3]) - 0.1)
        flipped = cv2.flip(right_image.copy(), 1)
        images_right.append(toRGB(flipped))
        measurements.append(-(float(line[3]) - 0.1))

    images = images_center + images_left + images_right
    return np.array(images), np.array(measurements)


def get_cropped_data(data):
    cropped = []
    for img in data:
        cropped.append(img[55:140, :, :])
    return np.array(cropped)
