import tensorflow as tf
import numpy as np
from utils import get_logger
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten


LOGGER = get_logger(__name__)
FC_LAYER_SIZE = 256


