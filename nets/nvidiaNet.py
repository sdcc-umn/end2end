import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

def nvidiaNet(x):
      
   conv1 = tcl.conv2d(x, 24, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv1')
   conv1 = tcl.batch_norm(conv1)
   conv1 = lrelu(conv1)
   
   conv2 = tcl.conv2d(conv1, 36, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv2')
   conv2 = tcl.batch_norm(conv2)
   conv2 = lrelu(conv2)
   
   conv3 = tcl.conv2d(conv2, 48, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')
   conv3 = tcl.batch_norm(conv3)
   conv3 = lrelu(conv3)
   
   conv4 = tcl.conv2d(conv3, 64, 3, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')
   conv4 = tcl.batch_norm(conv4)
   conv4 = lrelu(conv4)
   
   print 'conv5:',conv5
   return conv5


