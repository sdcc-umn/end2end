import h5py
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt

batch_size = 32
STACK=3

if __name__ == "__main__":
    subtract_mean = False
    hdf5_file = h5py.File('lane_dataset_thresholded_3stack.hdf5', 'r')
    if subtract_mean:
        mm = hdf5_file["train_mean"][0, ...]
        mm = mm[np.newaxis, ...]
    data_num = hdf5_file["train_img"].shape[0]
    
    batches_list = list(range(int(ceil(float(data_num) / batch_size))))
    shuffle(batches_list)
    # loop over batches
    for n, i in enumerate(batches_list):
        i_s = i * batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch
        # read batch images and remove training mean
        images = hdf5_file["train_img"][i_s:i_e, ...]
        if subtract_mean:
            images -= mm
            # read labels and convert to one hot encoding
        ctrl = hdf5_file["train_ctrl"][i_s:i_e]
        print(n+1, '/', len(batches_list))
        fig = plt.figure(figsize=(8, 8))
        for a in range(STACK):
            fig.add_subplot(1, STACK,a+1)
            plt.imshow(images[0][:, :, 0, a])
            plt.title("ctrl: {}".format(ctrl[0]))
        plt.show()
        if n == 10:  # break after 5 batches
            break
    hdf5_file.close()
