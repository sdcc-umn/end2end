import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys
from utils import get_logger

LOGGER = get_logger(__name__)

sys.path.insert(0, 'ops/')
from ops.tf_ops import *

def nvidiaNet(x):
   LOGGER.debug("input shape X is: {}", x.get_shape())
   conv1 = tcl.conv2d(x, 24, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv1')
   conv1 = tcl.batch_norm(conv1)
   conv1 = lrelu(conv1)
   
   conv2 = tcl.conv2d(conv1, 36, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv2')
   conv2 = tcl.batch_norm(conv2)
   conv2 = lrelu(conv2)
   
   conv3 = tcl.conv2d(conv2, 48, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')
   conv3 = tcl.batch_norm(conv3)
   conv3 = lrelu(conv3)
   
   conv4 = tcl.conv2d(conv3, 64, 3, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv4')
   conv4 = tcl.batch_norm(conv4)
   conv4 = lrelu(conv4)

   flat = tcl.flatten(conv4)
   fc1 = tcl.fully_connected(flat, 2)

   return fc1


def simple_cnn(x):
    FC_LAYER_SIZE = 1024
    with tf.variable_scope("Conv1"):
        k_conv1 = tf.Variable(tf.truncated_normal([6, 6, 1, 10], stddev=0.1), name="kernels_conv1")
        b_conv1 = tf.Variable(tf.constant(0, tf.float32, [10]))
        conv1 = tf.nn.relu(tf.nn.conv2d(x, k_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)


    with tf.variable_scope("Conv2"):
        k_conv2 = tf.Variable(tf.truncated_normal([4, 4, 10, 14], stddev=0.1), name="kernels_conv2")
        b_conv2 = tf.Variable(tf.constant(0, tf.float32, [14]))
        conv2 = tf.nn.elu(tf.nn.conv2d(conv1, k_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

    with tf.variable_scope("Conv3"):
        k_conv3 = tf.Variable(tf.truncated_normal([4, 4, 14, 18], stddev=0.1), name="kernels_conv3")
        b_conv3 = tf.Variable(tf.constant(0, tf.float32, [18]))
        conv3 = tf.nn.elu(tf.nn.conv2d(conv2, k_conv3, strides=[1, 2, 2, 1], padding='SAME') + b_conv3)

    with tf.variable_scope("Conv4"):
        k_conv4 = tf.Variable(tf.truncated_normal([3, 3, 18, 22], stddev=0.1), name="kernels_conv4")
        b_conv4 = tf.Variable(tf.constant(0, tf.float32, [22]))
        conv4 = tf.nn.elu(tf.nn.conv2d(conv3, k_conv4, strides=[1, 2, 2, 1], padding='SAME') + b_conv4)

    with tf.variable_scope("fc1"):
        newshape = int(np.prod(conv4.get_shape()[1:]))
        reshaped_conv = tf.reshape(conv4, [-1, newshape])
        #dropout = tf.placeholder(tf.float32, [], name="dropout")
        w_fc1 = tf.Variable(tf.truncated_normal([newshape, FC_LAYER_SIZE], stddev=0.1), name="weights_fc1")
        b_fc1 = tf.Variable(tf.constant(0, tf.float32, [FC_LAYER_SIZE]))
        fc1_nodropout = tf.nn.elu(tf.matmul(reshaped_conv, w_fc1) + b_fc1)
        fc1 = tf.nn.dropout(fc1_nodropout, 0.95)

    with tf.variable_scope("fc2"):
        newshape = int(np.prod(conv4.get_shape()[1:]))
        reshaped_conv = tf.reshape(conv4, [-1, newshape])
        #dropout = tf.placeholder(tf.float32, [], name="dropout")
        w_fc1 = tf.Variable(tf.truncated_normal([newshape, FC_LAYER_SIZE], stddev=0.1), name="weights_fc1")
        b_fc1 = tf.Variable(tf.zeros([FC_LAYER_SIZE]))
        fc1_nodropout = tf.nn.elu(tf.matmul(reshaped_conv, w_fc1) + b_fc1)
        fc1 = tf.nn.dropout(fc1_nodropout, .95)



    with tf.variable_scope("Output_Layer"):
        w_output = tf.Variable(tf.truncated_normal([FC_LAYER_SIZE, 2], stddev=0.1), name="weights_conv")
        b_output = tf.Variable(tf.constant(0, tf.float32, [2]))
        pred = tf.matmul(fc1, w_output) + b_output

    return pred
