from __future__ import print_function, division
import six
import scipy.misc as misc
import tensorflow as tf
import numpy as np
import argparse
import random
import ntpath
import time
import nets.nvidiaNet as nets
import sys

from utils import get_logger

LOGGER = get_logger(__name__)

# import our own stuff
sys.path.insert(0, 'ops/')
from ops.tf_ops import *
from ops.data_ops import *


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',type=int,default=32)
   # parser.add_argument('--DATA_DIR',   required=True,help='~/SDCC/data')
   parser.add_argument('--TRIAL',      required=True,help='Trial number', type=int)
   parser.add_argument('--NETWORK',    required=False,help='The network to use', default='simple_cnn')
   parser.add_argument('--EPOCHS',     required=False,help='How long to train',type=int,default=10)
   parser.add_argument('--STACK',      required=False,help='Number of frames to stack',type=int,default=4)
   a = parser.parse_args()

   BATCH_SIZE = a.BATCH_SIZE
   # DATA_DIR   = a.DATA_DIR
   NETWORK    = a.NETWORK
   EPOCHS     = a.EPOCHS
   #STACK      = a.STACK
   STACK = 1
   TRIAL      = a.TRIAL

   CHECKPOINT_DIR = 'checkpoints/NETWORK_'+NETWORK+'/STACK_'+str(STACK)+'/TRIAL_'+str(TRIAL)
   
   try: os.makedirs(CHECKPOINT_DIR)
   except: pass
   # step counter
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # placeholders for data going into the network
   images  = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 96, 128, STACK), name='images')
   control = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2), name='control')

   if NETWORK == 'nvidiaNet':
      pred = nets.nvidiaNet(images)
   elif NETWORK == 'simple_cnn':
      pred = nets.simple_cnn(images)
   else:
      LOGGER.error("No recognized network. exiting")
      sys.exit(-1)

   loss = tf.losses.mean_squared_error(control, pred)

   # tensorboard summaries
   tf.summary.scalar('loss', loss)
   merged_summary_op = tf.summary.merge_all()

   # Optimizer
   train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

   # Saver
   saver = tf.train.Saver(max_to_keep=3)

   # initialize session and global variables
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   # save graph.
   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      LOGGER.info("Restoring previous model...")
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         LOGGER.info("Model restored")
      except:
         LOGGER.info("Could not restore model")
         pass

   ## Begin training
   step = sess.run(global_step)
   batch = 1
   b = batchGenerator("trial_2", n_stack=STACK, inf=False)

   for epoch_c in range(EPOCHS):
      E = epochGenerator(b)
      for batch_images, batch_control in E:
          batch_images, batch_control = next(b)

          # run train op
          sess.run(train_op, feed_dict={images:batch_images, control:batch_control})

          # now get all losses and summary *without* performing a training step - for tensorboard
          loss_batch, summary = sess.run([loss, merged_summary_op], feed_dict={images:batch_images, control:batch_control})
          summary_writer.add_summary(summary, step)

          print('epoch:',epoch_c,'step:',step,'batch_loss:',loss_batch)
          step += 1

          if step%500 == 0:
             LOGGER.info('Saving model...')
             saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
             saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
             LOGGER.info('Done saving')

          batch+=1
   sess.close()
