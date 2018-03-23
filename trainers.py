import tensorflow as tf
from tqdm import tqdm
import numpy as np
from utils import get_logger

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
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for it in loop:
            loss = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        self.writer.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess) 

    def train_step(self):
        batch_x, batch_y = next(self.data.next_train_batch(self.config.batch_size))
        feed_dict = {self.model.images: batch_x, self.model.control: batch_y, self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_op, self.model.loss],
                                     feed_dict=feed_dict)
        print("loss: ", loss)
        return loss
