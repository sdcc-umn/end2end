from __future__ import print_function, division
import tensorflow as tf
import utils
from dataloader import DataGenerator
from model import simple_cnn
import trainers
import matplotlib.pyplot as plt
import random

LOGGER = utils.get_logger(__name__)


def main():
    try:
        args = utils.get_args()
        config = utils.process_config(args.config)
    except:
        print("Invalid arguments")
        exit(0)

    utils.create_dirs([config.summary_dir, config.checkpoint_dir])
    sess = tf.Session()
    model = simple_cnn(config)
    model.load(sess)  # if extant
    data = DataGenerator(config)
    writer = utils.ModelSummarizer(sess, config)
    trainer = trainers.BasicAdamTrainer(sess, model, data, config, writer)

   # trainer.train()
    loss = trainer.test()
    print("Avg loss: {}".format(loss/data.len_test))

    fig, axs = plt.subplots(1, 5)
    for i in range(3):
        j = random.randint(0, data.len_test)
        image = data.dataset_file["test_img"][j, ...]
        image = image[..., -1]
        ctrl = data.dataset_file["test_ctrl"][j]
        pred = model.predict(image, sess)
        axs[i].imshow(image[:, :, 0])
        axs[i].set_title("true: {}\npred: {:.2f}".format(ctrl, pred[0]))
    plt.show()

    sess.close()


if __name__ == "__main__":
    main()
