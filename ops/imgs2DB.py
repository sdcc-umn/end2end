import sys
import os
import random
import logging
import h5py
import numpy as np
import cv2
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


'''
Take all of the SDCC dataset files and convert them into a train/test dataset.


esizing all images in --src folder and saves in --dest folder
------- speficy image size in --w and --h
'''

WIDTH = 224
HEIGHT = 224
THRESHOLD = True
N_CHANNELS = 1 if THRESHOLD else 3
N_STACK = 1
STRIDE = 1
TEST = True

def preprocess_img(img_path, threshold=THRESHOLD):
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([100, 200, 100], dtype="uint8")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if threshold:
        mask2 = cv2.inRange(img, lower, upper)
        _, thresholded = cv2.threshold(mask2, 20, 255, 0)
        thresholded = np.expand_dims(thresholded, axis=-1)
        return thresholded
    return img


def preprocess_stack(img_path_list, N_STACK):
    stacked_imgs = np.zeros(shape=(1, HEIGHT, WIDTH, N_CHANNELS, N_STACK))
    for i in range(N_STACK):
        cur_img = preprocess_img(img_path_list[i])
        stacked_imgs[..., i] = cur_img
    return stacked_imgs


# output will be a dataset called lane_dataset.hdf5
# step 1: get a list of all the images and their associated control
# tuples (from the annotations files)
trial_set = ['trial_1', 'trial_2']
stacked_img_control_pairs = []
for trial in trial_set:
    path_to_annotations = os.path.join(trial, "annotations.txt")
    LOGGER.info("Path to annotations: {}".format(path_to_annotations))
    with open(path_to_annotations, 'r') as annot_file:
        img_control_pairs = annot_file.readlines()

    # step 2: pad the images into 3 imgs per control output
    for stack_idx in range(0, len(img_control_pairs)-N_STACK, STRIDE):
        stacked_pair = []
        for offset_idx in range(N_STACK-1):
            # we only want the last image's control tuple;
            # otherwise just take img.
            img = os.path.join(trial, "images", img_control_pairs[stack_idx + offset_idx].split(" ")[0])
            stacked_pair.append(img)
        # on the last item, we want both the image and the control tuple
        final_split = img_control_pairs[stack_idx + N_STACK - 1].split(' ')
        img = os.path.join(trial, "images", final_split[0])
        ctrl_tuple = np.array(list(map(lambda x: float(x.strip()), final_split[1:])))
        # now combine the N_STACK worth of images and the control tuple:
        stacked_pair.append(img)
        stacked_pair.append(ctrl_tuple)
        stacked_img_control_pairs.append(stacked_pair)
        LOGGER.debug(str(stacked_pair))

# step 3: now we should have sets of [img1, img2, img3, (ctrl tuple)]. shuffle these
random.shuffle(stacked_img_control_pairs)
# step 4: divide these into train, validation, and test splits: 60, 20, 20
LOGGER.info("entire dataset: {}".format(len(stacked_img_control_pairs)))
idx1 = int(0.6 * len(stacked_img_control_pairs))
idx2 = int(0.5 * (len(stacked_img_control_pairs) - idx1))
LOGGER.debug("idx1: {0}\t idx2: {1}".format(idx1, idx2))
train = stacked_img_control_pairs[:idx1]
val = stacked_img_control_pairs[idx1:idx1+idx2]
test = stacked_img_control_pairs[idx1+idx2:]
LOGGER.info('train: {}\tval: {}\ttest: {}'.format(len(train), len(val), len(test)))

# Step 5: save these sets organized in a h5py database
train_shape = (len(train), WIDTH, HEIGHT, N_CHANNELS, N_STACK)
val_shape = (len(val), WIDTH, HEIGHT, N_CHANNELS, N_STACK)
test_shape = (len(test), WIDTH, HEIGHT, N_CHANNELS, N_STACK)

# Initialize/pre-allocate h5 database
dataset_output_path = 'lane_dataset.hdf5'
hdf5_file = h5py.File(dataset_output_path, mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.uint8)
hdf5_file.create_dataset("val_img", val_shape, np.uint8)
hdf5_file.create_dataset("test_img", test_shape, np.uint8)
hdf5_file.create_dataset("train_mean", train_shape, np.float32)
hdf5_file.create_dataset("train_ctrl", (len(train),2), np.float32)
hdf5_file.create_dataset("val_ctrl", (len(val),2), np.float32)
hdf5_file.create_dataset("test_ctrl", (len(test),2), np.float32)

# Now actually populate the database
# mean = np.zeros(train_shape[1:], np.float32)
mean = np.zeros(train_shape, np.float32)

# loop over train addresses
LOGGER.info("Adding train data to dataset")
for i in tqdm(range(len(train))):
    stack = train[i]
    ctrl_tuple = stack[-1]
    stack = preprocess_stack(stack[:-1], N_STACK)
    assert stack.shape == (1, WIDTH, HEIGHT, N_CHANNELS, N_STACK), "expected {} got {}".format((1, WIDTH, HEIGHT, N_CHANNELS, N_STACK), stack.shape)
    hdf5_file["train_img"][i, ...] = stack  # img[None]
    hdf5_file["train_ctrl"][i, ...] = ctrl_tuple
    mean += stack / len(train)

LOGGER.info("Adding val data to dataset")
for i in tqdm(range(len(val))):
    stack = train[i]
    ctrl_tuple = stack[-1]
    stack = preprocess_stack(stack[:-1], N_STACK)
    assert stack.shape == (1, WIDTH, HEIGHT, N_CHANNELS, N_STACK), "expected {} got {}".format((1, WIDTH, HEIGHT, N_CHANNELS, N_STACK), stack.shape)

    hdf5_file["val_img"][i, ...] = stack
    hdf5_file["val_ctrl"][i, ...] = ctrl_tuple


LOGGER.info("Adding test data to dataset")
for i in tqdm(range(len(test))):
    stack = test[i]
    ctrl_tuple = stack[-1]
    stack = preprocess_stack(stack[:-1], N_STACK)
    assert stack.shape == (1, WIDTH, HEIGHT, N_CHANNELS, N_STACK), "expected {} got {}".format((1, WIDTH, HEIGHT, N_CHANNELS, N_STACK), stack.shape)
    hdf5_file["test_img"][i, ...] = stack
    hdf5_file["test_ctrl"][i, ...] = ctrl_tuple


# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()


class ProcessImg(object):
    def __init__(self):
        self.im_ext_ = ['jpg', 'jpeg', 'bmp', 'png', 'tiff', 'ppm', 'pgm']
        self.lower = np.array([0, 0, 0], dtype="uint8")
        self.upper = np.array([100, 200, 100], dtype="uint8")

    def check_file_ext(self, f_name):
        for ext_ in self.im_ext_:
            if f_name.lower().endswith(ext_):
                return True
            return False

    def process_all(self, src, dest):
            if self.check_file_ext(im_file):
                # Convert to HSV color space
                hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                # Create a binary thresholded image
