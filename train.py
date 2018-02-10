import scipy.misc as misc
import tensorflow as tf
import numpy as np
import argparse
import random
import ntpath
import time
import sys
import cv2
import os

# import our own stuff
sys.path.insert(0, 'ops/')
from tf_ops import *
from data_ops import *


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',type=int,default=64)
   parser.add_argument('--DATA_DIR',   required=True,help='Directory where data is')
   parser.add_argument('--TRIAL',      required=True,help='Trial number', type=int)
   parser.add_argument('--NETWORK',    required=False,help='The network to use', default='nvidiaNet')
   parser.add_argument('--EPOCHS',     required=False,help='How long to train',type=int,default=100)
   parser.add_argument('--STACK',      required=False,help='Number of frames to stack',type=int,default=4)
   a = parser.parse_args()

   BATCH_SIZE = a.BATCH_SIZE
   DATA_DIR   = a.DATA_DIR
   NETWORK    = a.NETWORK
   EPOCHS     = a.EPOCHS
   STACK      = a.STACK
   TRIAL      = a.TRIAL

   CHECKPOINT_DIR = 'checkpoints/NETWORK_'+NETWORK+'/STACK_'+str(STACK)+'/TRIAL_'+str(TRIAL)
   
   try: os.makedirs(CHECKPOINT_DIR)
   except: pass

   data = loadData(DATA_DIR, TRIAL)

   # step counter
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # placeholders for data going into the network
   images  = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, STACK), name='images')
   control = tf.placeholder(tf.float32, shape=(BATCH_SIZE, STACK*3), name='control')

   if NETWORK == 'nvidiaNet':
      from nvidiaNet import nvidiaNet
      pred = nvidiaNet(images)

   loss = tf.nn.l2_loss(pred-control)

   # tensorboard summaries
   tf.summary.scalar('loss', loss)
   merged_summary_op = tf.summary.merge_all()

   train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion

   step = sess.run(global_step)

   epoch_num = step/(train_len/BATCH_SIZE)

   while epoch <= EPOCHS:
      
      epoch_num = step/(train_len/BATCH_SIZE)
      start = time.time()

      # run train op
      sess.run(train_op, feed_dict={images:batch_images, control:batch_control})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([loss, merged_summary_op], feed_dict={images:batch_images, control:batch_control})
      summary_writer.add_summary(summary, step)

      print 'epoch:',epoch,'step:',step,'loss:',loss
      step += 1
    
      if step%500 == 0:
         print
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Done saving'
         print
