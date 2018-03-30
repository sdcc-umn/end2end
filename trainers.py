import tensorflow as tf
from tqdm import tqdm
import numpy as np
from utils import get_logger
from tflearn.helpers.trainer import Trainer

LOGGER = get_logger(__name__)


class BaseTrain:
    def __init__(self, sess, model, data, config, writer):
        self.model = model
        self.writer = writer
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError


class BasicAdamTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, writer):
        super(BasicAdamTrainer, self).__init__(sess, model, data, config, writer)

    def train_epoch(self):
        loop = tqdm(range(1, self.config.num_iter_per_epoch))
        train_losses = []
        for it in loop:
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if it %500 ==0:
                val_loss = self.val()
                cur_it = self.model.global_step.eval(self.sess)
                summaries_dict = {}
                summaries_dict['val_loss'] = val_loss
                summaries_dict['train_loss'] = np.mean(train_loss)
                train_losses = []
                self.writer.summarize(cur_it, summaries_dict=summaries_dict)
                self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_train_batch(self.config.batch_size))
        feed_dict = {self.model.images: batch_x, self.model.control: batch_y, self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_op, self.model.loss],
                                     feed_dict=feed_dict)
        return loss

    def val(self):
        losses = []
        for i in range(self.data.n_val_batches):
            batch_x, batch_y = next(self.data.next_val_batch(self.config.batch_size))
            feed_dict = {self.model.images: batch_x, self.model.control: batch_y, self.model.is_training: False}
            loss = self.sess.run([self.model.loss], feed_dict=feed_dict)
            losses.append(loss)
        return np.mean(loss)

    def test(self):
        losses = []
        for i in range(self.data.n_train_batches):
            batch_x, batch_y = next(self.data.next_train_batch(self.config.batch_size))
            feed_dict = {self.model.images: batch_x, self.model.control: batch_y, self.model.is_training: False}
            loss = self.sess.run([self.model.loss], feed_dict=feed_dict)
            losses.append(loss)
        return np.mean(loss)






