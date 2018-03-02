'''

Operations used for data management

'''

from scipy import misc
import tensorflow as tf
import numpy as np
import math
import time
import random
import glob
import os
import fnmatch

# [-1,1] -> [0, 255]
def deprocess(x):
   return (x+1.0)*127.5

# [0,255] -> [-1, 1]
def preprocess(x):
   return (x/127.5)-1.0

def get_image(img_path):
   return np.ones((20,20))

def get_control_tuple(line):
   return np.array([0,0])

def batchGenerator(DATA_DIR, batch_size=32, n_stack=1, bw = False):
   """ A genrator that returns a batch of data, parings: [Batch_Size, stack, image] """

   # get the path name for the annotations file, which contains (img, C) pairs.
   annot_file = os.path.join(DATA_DIR, 'annotation.txt')
   # get path for the images directory
   imgdir = os.path.join(DATA_DIR, 'images')

   with open(annot_file, 'r') as annot:
      # X is a stacked img array, of shape [batch_size, width, height, n_stack]
      width = 20
      height = 20
      while True:
         X = np.ones((1, 20,20,1))
         Y = np.ones((1, 2))
         # Y is [batch_size, 2]
         for _ in range(batch_size):
            sample = np.zeros((width, height, 1))

            for _ in range(n_stack):
               # get image path from annot file
               line=annot.readline()
               if line == '':
                  annot.seek(0)
                  line = annot.readline()
               assert(line!='')
               img_path = annot.readline().split(" ")[0]
               # get image of [width, height]
               img = np.expand_dims(get_image(img_path), axis=-1)
               # stack this along the n_stack dimension
               sample = np.concatenate((sample, img), axis=-1)
               y = get_control_tuple(line)
            X = np.concatenate((X, np.expand_dims(sample[:, :, 1:], axis=0)), axis=0)
            Y = np.concatenate((Y, np.expand_dims(y, axis=0)), axis=0)
         X = X[1:]
         Y = Y[1:]
         assert X.shape == (batch_size, width, height, n_stack), "expected {}\t got {}".format([batch_size+1, width, height, n_stack+1], X.shape)
         assert Y.shape == (batch_size, 2), Y.shape
         yield X, Y

