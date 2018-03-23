from __future__ import print_function, division
import tensorflow as tf
import utils
from dataloader import DataGenerator
from model import simple_cnn
import trainers

LOGGER = utils.get_logger(__name__)


def main():
    try:
        args = utils.get_args()
        config = utils.process_config(args.config)
    except:
        print("Invalid arguments")
        exit(0)

    # create the experiments dirs
    utils.create_dirs([config.summary_dir, config.checkpoint_dir])

    sess = tf.Session()
    model = simple_cnn(config)
    model.load(sess)  # if extant
    data = DataGenerator(config)
    writer = utils.ModelSummarizer(sess, config)
    trainer = trainers.BasicAdamTrainer(sess, model, data, config, writer)

    # here you train your model
    trainer.train()

    sess.close()


if __name__ == "__main__":
    main()
