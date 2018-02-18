N_CLASSES = 10
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 2
N_HIDDEN_VARS= 256

with tf.name_scope('misc'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, 784], name="X")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y")
    images = tf.reshape(X, shape=[-1, 28, 28])
    images = tf.unstack(value=images, axis=1, name="unstack")

with tf.variable_scope("lstm1"):
    lstm1 = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_VARS)
    lstm1_outputs, lstm1_states = tf.nn.static_rnn(lstm1, images, dtype=tf.float32)


with tf.variable_scope("output_layer"):
    w_output = tf.Variable(tf.random_normal([N_HIDDEN_VARS, 10]))
    b_output = tf.Variable(tf.random_normal([10]))
    tf_logits= tf.matmul(lstm1_states[1], w_output)+b_output
    sofmaxed = tf_softmaxed = tf.nn.softmax(tf_logits)

with tf.variable_scope("Metrics"):
    tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf_logits, labels=Y))
    tf.summary.scalar('batch loss', tf_loss)
    tf_correct = tf.equal(tf.argmax(tf_softmaxed, 1), tf.argmax(Y, 1))
    tf_accuracy = tf.reduce_mean(tf.cast(tf_correct, tf.float32))
    tf.summary.scalar('batch accuracy', tf_accuracy)

with tf.variable_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(tf_loss, global_step=global_step)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    # to visualize using TensorBoard
    graph_folder = './graph/mnist_lstm'
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)

    writer = tf.summary.FileWriter(graph_folder, sess.graph)
    initial_step = global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    for index in range(initial_step, int(n_batches * N_EPOCHS)):  # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, {X: X_batch, Y:Y_batch})
        if (index+1) % SKIP_STEP == 0:
            merged_summary, batch_accuracy, batch_loss = sess.run([merged, tf_accuracy, tf_loss],
                                      feed_dict={X: X_batch, Y: Y_batch})
            writer.add_summary(merged_summary, index)
            print("{0} -- batch_accuracy={1:.5f} -- batch_loss={2:.4f}".format(index, batch_accuracy, batch_loss))

    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))

    # test the model
    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    X_batch, Y_batch = mnist.test.next_batch(mnist.test.num_examples)
    total_accuracy = sess.run(tf_accuracy, feed_dict={X:X_batch, Y:Y_batch})

    print("Total Test-Set Accuracy: {0}".format(total_accuracy))
