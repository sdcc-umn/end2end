import tensorflow as tf
import numpy as np
from utils import get_logger

LOGGER = get_logger(__name__)
FC_LAYER_SIZE = 1024


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()

    def _save_model_checkpoint(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    def _load_model_checkpoint(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    def load(self, sess):
        self._load_model_checkpoint(sess)

    def init_cur_epoch(self):
        """inialize a tensorflow variable to use as epoch counter"""
        with tf.variable_scope('cur_epoch_counter'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        """ inialize a tensorflow variable to use as a global step counter"""
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step= tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        raise NotImplementedError


class simple_cnn(BaseModel):
    def __init__(self, config):
        super(simple_cnn, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.FC_LAYER_SIZE = 1024
        self.is_training = tf.placeholder(tf.bool)
        self.images = tf.placeholder(tf.float32, shape=[None] + self.config.input_shape, name='images')
        self.control = tf.placeholder(tf.float32, shape=[None] + self.config.output_shape, name='control')

        with tf.variable_scope("Conv1"):
            k_conv1 = tf.Variable(tf.truncated_normal([6, 6, 1, 10], stddev=0.1), name="kernels_conv1")
            b_conv1 = tf.Variable(tf.constant(0, tf.float32, [10]))
            conv1 = tf.nn.relu(tf.nn.conv2d(self.images, k_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

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
            # dropout = tf.placeholder(tf.float32, [], name="dropout")
            w_fc1 = tf.Variable(tf.truncated_normal([newshape, FC_LAYER_SIZE], stddev=0.1), name="weights_fc1")
            b_fc1 = tf.Variable(tf.zeros([FC_LAYER_SIZE]))
            fc1_nodropout = tf.nn.elu(tf.matmul(reshaped_conv, w_fc1) + b_fc1)
            fc1 = tf.nn.dropout(fc1_nodropout, .95)

        with tf.variable_scope("Output_Layer"):
            w_output = tf.Variable(tf.truncated_normal([FC_LAYER_SIZE]+self.config.output_shape, stddev=0.1), name="weights_conv")
            b_output = tf.Variable(tf.constant(0, tf.float32, self.config.output_shape))
            self.pred = tf.matmul(fc1, w_output) + b_output

        with tf.variable_scope("loss"):
            self.loss = tf.losses.mean_squared_error(self.control, self.pred)

        with tf.variable_scope("optimizer"):
            self.train_op = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate).minimize(self.loss, global_step = self.global_step)


    def predict(self, img, sess=None):
        return sess.run(self.pred(img))
