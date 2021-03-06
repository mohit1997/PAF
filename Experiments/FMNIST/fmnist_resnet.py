import argparse
import sys
import tempfile

import tensorflow as tf
import numpy as np
from imgaug import augmenters as iaa
from utils import *

np.random.seed(0)
tf.set_random_seed(0)

def lr(iteration):
    if iteration<7000:
        return 1e-3
    if iteration<12000:
        return 1e-4
    return 1e-5
def res_block(x, filters, is_train):
    shortcut = x

    h = x

    h = tf.layers.conv2d(h, filters=filters, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.layers.batch_normalization(h, training=is_train)
    h = tf.nn.relu(h)

    h = tf.layers.conv2d(h, filters=filters, kernel_size=(1, 1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.layers.batch_normalization(h, training=is_train)

    h = h + shortcut
    h = tf.nn.relu(h)

    return h

def res_block2(x, filters, is_train):
    shortcut = x
    h = x

    h = tf.layers.conv2d(h, filters=filters, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.layers.batch_normalization(h, training=is_train)
    h = tf.nn.relu(h)

    h = tf.layers.conv2d(h, filters=filters, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.layers.batch_normalization(h, training=is_train)

    h = h + tf.layers.conv2d(shortcut, filters=filters, kernel_size=(1, 1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.nn.relu(h)

    return h



def deepnn(x,n_classes):
    is_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    # h = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), padding='SAME')

    h = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.layers.batch_normalization(h, training=is_train)
    h = tf.nn.relu(h)

    h = tf.layers.conv2d(h, filters=32, kernel_size=(1, 1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.layers.batch_normalization(h, training=is_train)
    # h = tf.nn.relu(h)


    # h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2))
    h = tf.contrib.layers.avg_pool2d(h, kernel_size=(2, 2), stride=(2, 2))

    h = res_block2(h, 64, is_train)
    
    # h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2))
    h = tf.contrib.layers.avg_pool2d(h, kernel_size=(2, 2), stride=(2, 2))

    h = res_block2(h, 128, is_train)

    # h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2))
    h = tf.contrib.layers.avg_pool2d(h, kernel_size=(2, 2), stride=(2, 2))

    h = tf.contrib.layers.flatten(h)

    h = tf.contrib.layers.fully_connected(h, 512, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.layers.batch_normalization(h, training=is_train)
    h = tf.nn.relu(h)

    # h = tf.contrib.layers.fully_connected(h, 512, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.contrib.layers.fully_connected(h, 512, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005))
    h = tf.layers.batch_normalization(h, training=is_train)
    h = tf.nn.relu(h)    
    h = tf.contrib.layers.fully_connected(h, n_classes, activation_fn=None)

    return h, is_train, keep_prob


def main(_):
    # Import data
    mnist = FMNIST_aug()
    mnist.data_augmentation()
    n_classes = 10

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    learning_rate = tf.placeholder(tf.float32)
    # Build the graph for the deep net
    y_conv, is_train, keep_prob = deepnn(x, n_classes)

    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
        reg_loss = tf.losses.get_regularization_loss()

    with tf.name_scope('Adam_optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_step = tf.contrib.opt.AdamWOptimizer(0.1*learning_rate, learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
            reg_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(reg_loss)

            train_step = tf.group(train_step, reg_step)

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    acc = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(15000):
            batch = mnist.next_train_batch(64, augment=False)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], is_train: True, keep_prob: 0.8})
                a = accuracy.eval(feed_dict={x: mnist.get_test_images(), y_: mnist.get_test_labels(), is_train: False, keep_prob: 1.0})
                print('Step %d, Training accuracy %g, Testing accuracy %g' % (i, train_accuracy, a))
                acc.append(a)

            train_step.run(feed_dict={x: batch[0], y_: batch[1], is_train: True, learning_rate: lr(i), keep_prob: 0.8})

        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.get_test_images(), y_: mnist.get_test_labels(), is_train: False, keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)