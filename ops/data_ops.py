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
import cPickle as pickle

# [-1,1] -> [0, 255]
def deprocess(x):
   return (x+1.0)*127.5

# [0,255] -> [-1, 1]
def preprocess(x):
   return (x/127.5)-1.0

def loadData(DATA_DIR, TRIAL):

   images = glob.glob(DATA_DIR+'trial_'+str(TRIAL)+'/images/*.jpg')

   d = {}

   with open(DATA_DIR+'trial_'+str(TRIAL)+'/annotation.txt', 'r') as f:
      for line in f:
         line = line.rstrip().split()

         control = np.asarray([0, 0, 0]).astype('float32')
         if line[1] == 'v': control[0] = 1
         else: control[1] = 1

         control[2] = float(line[2])
         d[line[0]] = control

   return d

