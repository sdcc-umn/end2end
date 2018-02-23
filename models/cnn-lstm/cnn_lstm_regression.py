#!/usr/bin/python3
"""CNN - LSTM regression"""
import numpy as np
import logging
from base.base_model import BaseModel
import tensorflow as tf
import os
import time
from tensorflow.examples.tutorials.mnist import mnist
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

LOGGER = logging.getLogger("CnnLSTM")
N_LATENT_VARS = 1024
BATCH_SIZE = 16
N_EPOCHS = 10
SKIP_STEP = 1

class CNNLSTM_R_Model(BaseModel):
    def __init__(self, config):
        super(CNNLSTM_R_Model, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # tf graph and loss function construction
        with tf.name_scope('misc'):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None, 784], name="X")
            self.Y = tf.placeholder(tf.float32, [None, 10], name="Y")
            images = tf.reshape(self.X, shape=[-1, 28, 28])
            images = tf.unstack(value=images, axis=1, name="unstack")

        with tf.variable_scope("lstm"):
            lstm1 = tf.contrib.rnn.BasicLSTMCell(N_LATENT_VARS)
            lstm1_outputs, lstm1_states = tf.nn.static_rnn(lstm1, images, dtype=tf.float32)


        with tf.variable_scope("output_layer"):
            # Output layer should contain a regressed value for a and v
            w_output = tf.Variable(tf.random_normal([N_LATENT_VARS, 2]))
            b_output = tf.Variable(tf.zeros([2])) # bias towards zero, as it will be likely that we won't want a bias here.
            tf_logits= tf.matmul(lstm1_states[1], w_output)+b_output
            # no activation on output

        with tf.variable_scope("Metrics"):
            self.loss = tf.reduce_mean(tf.squared_difference(logits = tf_logits, labels=self.Y))
            tf.summary.scalar('batch loss', self.loss)
            tf_correct = tf.equal(tf_logits, self.Y)
            self.accuracy = tf.reduce_mean(tf.cast(tf_correct, tf.float32))
            tf.summary.scalar('batch accuracy', self.accuracy)

        with tf.variable_scope("Optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            merged = tf.summary.merge_all()

            LOGGER.debug("writing graph")
            graph_folder = './graph'
            if not os.path.exists(graph_folder):
                os.makedirs(graph_folder)
            writer = tf.summary.FileWriter(graph_folder, sess.graph)

            initial_step = self.global_step.eval()

            start_time = time.time()
            n_batches = int(mnist.train.num_examples / BATCH_SIZE)

            for index in range(initial_step, int(n_batches * N_EPOCHS)):  # train the model n_epochs times
                X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
                sess.run(self.optimizer, {self.X: X_batch, self.Y: Y_batch})
                if (index + 1) % SKIP_STEP == 0:
                    merged_summary, batch_accuracy, batch_loss = sess.run([merged, self.tf_accuracy, self.tf_loss],
                                                                          feed_dict={self.X: X_batch, self.Y: Y_batch})
                    writer.add_summary(merged_summary, index)
                    LOGGER.info(
                        "{0} -- batch_accuracy={1:.5f} -- batch_loss={2:.4f}".format(index, batch_accuracy, batch_loss))

            LOGGER.info("Optimization Finished!")  # should be around 0.35 after 25 epochs
            LOGGER.info("Total time: {0} seconds".format(time.time() - start_time))

    def saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def test(self):
        with tf.Session() as sess:
            n_batches = int(mnist.test.num_examples / BATCH_SIZE)
            X_batch, Y_batch = mnist.test.next_batch(mnist.test.num_examples)
            total_accuracy = sess.run(self.tf_accuracy, feed_dict={X:X_batch, Y:Y_batch})

        LOGGER.info("Total Test-Set Accuracy: {0}".format(total_accuracy))
