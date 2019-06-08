import argparse
import sys
import tempfile

import tensorflow as tf
import numpy as np
from taylor_batch_norm import taylor
from imgaug import augmenters as iaa
from utils import *

np.random.seed(0)
tf.set_random_seed(0)

num_features = 1000
batchsize = 2048

# For degree 4
def lr(iteration):
    if iteration<200:
        return 1e-2
    if iteration<1000:
        return 1e-3
    if iteration<2000:
        return 1e-4
    return 1e-4

def gaussian_noise_layer(input_layer, is_train, std):
    noise = tf.cond(is_train, lambda: input_layer + tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32), lambda: input_layer)
    return noise


def deepnn(x, n_classes):
    is_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    h = tf.contrib.layers.fully_connected(x, 256, activation_fn=None, weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.0005))
    # h = tf.keras.layers.Dense(1000)(x)
    h = taylor(h, k=2, is_train=is_train, name="Ac/1")
    # h = tf.layers.batch_normalization(h, training=is_train)
    # h = tf.nn.tanh(h)

    h = tf.contrib.layers.fully_connected(h, 128, activation_fn=None)
    # h = tf.keras.layers.Dense(1000)(x)
    h = taylor(h, k=2, is_train=is_train, name="Ac/2")
    # h = tf.layers.batch_normalization(h, training=is_train)
    # h = tf.nn.tanh(h)

    h = tf.contrib.layers.fully_connected(h, 64, activation_fn=None)
    # h = tf.keras.layers.Dense(1000)(x)
    h = taylor(h, k=2, is_train=is_train, name="Ac/3")

    # h = tf.contrib.layers.fully_connected(h, 2048, activation_fn=None)
    # h = taylor(h, k=2, is_train=is_train, name="Ac/2")
    # h = tf.keras.layers.Dense(1)(h)
    h = tf.contrib.layers.fully_connected(h, n_classes, activation_fn=None)

    return h, is_train, keep_prob


def main(_):
    # Import data
    X, Y = get_polynomial(num_features=num_features, degree=4, samples=100000)
    print(X.shape, Y.shape)

    x_train, y_train, x_test, y_test = split(X, Y, train_split=0.7)

    print(x_train[:, 3:4] - y_train)

    # sys.exit()

    x = tf.placeholder(tf.float32, [None, num_features])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])

    learning_rate = tf.placeholder(tf.float32)
    # Build the graph for the deep net
    y_conv, is_train, keep_prob = deepnn(x, 1)

    with tf.name_scope('Loss'):
        def include_activation(name):
            return ('activation_coeff' in name)
        mean_squared_error = (y_ - y_conv)**2
        mean_squared_error = tf.reduce_mean(mean_squared_error)
        w_loss = tf.losses.get_regularization_loss()
        graph = tf.get_default_graph()
        temp = [op.values()[0] for op in graph.get_operations() if ((len(op.values()) >= 1) and (include_activation(op.values()[0].name)))]
        print(temp)
        # regl1_loss = 0.01 * tf.add_n([tf.reduce_sum(0.01* tf.abs(tf.cast(v, tf.float32))) for v in temp]),
        regl2_loss =  0.01 * tf.add_n([tf.reduce_sum(0.1*tf.nn.l2_loss(tf.cast(v, tf.float32))) for v in temp])
        # reg_loss = regl1_loss

    with tf.name_scope('Adam_optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_step = tf.contrib.opt.AdamWOptimizer(0.1*learning_rate, learning_rate).minimize(mean_squared_error)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_squared_error+w_loss)
            reg_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(regl2_loss)

            train_step = tf.group(train_step, reg_step)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    acc = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            for batch_x, batch_y in iterate_minibatches(x_train, y_train, batchsize):
                train_step.run(feed_dict={x: batch_x, y_: batch_y, is_train: True, learning_rate: lr(i), keep_prob: 0.8})
            if i % 10 == 0:
                train_accuracy, regloss = sess.run([mean_squared_error, regl2_loss+w_loss], feed_dict={x: x_train, y_: y_train, is_train: False, keep_prob: 0.8})
                a = mean_squared_error.eval(feed_dict={x: x_test, y_: y_test, is_train: False, keep_prob: 1.0})
                print('Step %d, Training Loss %g, Testing Loss %g, Regularization Loss %g' % (i, train_accuracy, a, regloss))
                acc.append(a)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)