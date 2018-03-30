import numpy as np
import h5py
from random import shuffle


class DataGenerator():
    def __init__(self, config):
        self.config = config
        self.datapath = config.datapath
        self.dataset_file = h5py.File(config.datapath, 'r')
        self.img_shape = self.dataset_file["train_img"][0].shape
        self.ctrl_shape = self.dataset_file["train_ctrl"][0].shape

        self.len_train = self.dataset_file["train_img"].shape[0]
        self.len_test = self.dataset_file["test_img"].shape[0]
        self.len_val = self.dataset_file["val_img"].shape[0]
        self.config["num_iter_per_epoch"] = self.len_train

        batch_size = config.batch_size
        self.n_train_batches = int(self.len_train/batch_size)
        self.n_val_batches = int(self.len_val / batch_size)
        self.n_test_batches = int(self.len_test / batch_size)
        batches = list(range(self.n_train_batches))
        shuffle(batches)
        self.batches = batches

    def next_train_batch(self, batch_size):
        for n, i in enumerate(self.batches):
            i_s = i * batch_size
            i_e = min([(i + 1) * batch_size, self.len_train])
            images = self.dataset_file["train_img"][i_s:i_e, ..., 0]
            ctrl = self.dataset_file["train_ctrl"][i_s:i_e]
            yield images, ctrl

    def next_val_batch(self, batch_size):
        for i in range(self.n_val_batches):
            i_s = i * batch_size
            i_e = min([(i + 1) * batch_size, self.len_val])
            images = self.dataset_file["val_img"][i_s:i_e, :, :, :, 0]
            ctrl = self.dataset_file["val_ctrl"][i_s:i_e]
            yield images, ctrl

    def next_test_batch(self, batch_size):
        for i in range(self.n_test_batches):
            i_s = i * batch_size
            i_e = min([(i + 1) * batch_size, self.len_test])
            images = self.dataset_file["test_img"][i_s:i_e, ..., 0]
            ctrl = self.dataset_file["test_ctrl"][i_s:i_e]
            yield images, ctrl
