import argparse
import sys
import tempfile

import tensorflow as tf
import numpy as np
from imgaug import augmenters as iaa
from utils import *
from taylor_batch_norm_backup import taylor
np.random.seed(0)
tf.set_random_seed(0)

clip = False

def lr(iteration):
    if iteration<3000:
        return 1e-3
    if iteration<7000:
        return 1e-3
    if iteration<12000:
        return 1e-4
    if iteration<16000:
        return 1e-5
    return 1e-4

def deepnn(x,n_classes):
    is_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    h1 = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    h = taylor(h1, k=2, is_train=is_train, name="Ac/1")

    h = tf.layers.conv2d(h, filters=32, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    h = taylor(h, k=2, is_train=is_train, name="Ac/2")
    h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2))
    # h = tf.contrib.layers.avg_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
    
    h1 = tf.layers.conv2d(h, filters=64, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    h = taylor(h1, k=2, is_train=is_train, name="Ac/3")
    # h = tf.contrib.nn.alpha_dropout(h, keep_prob=keep_prob)
    h = tf.layers.conv2d(h, filters=64, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    h = taylor(h, k=2, is_train=is_train, name="Ac/4") + h1
    h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2))

    h1 = tf.layers.conv2d(h, filters=128, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    h = taylor(h1, k=2, is_train=is_train, name="Ac/5")
    # h = tf.contrib.nn.alpha_dropout(h, keep_prob=keep_prob)
    h = tf.layers.conv2d(h, filters=128, kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    h = taylor(h, k=2, is_train=is_train, name="Ac/6") + h1
    h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2))
    # h = tf.contrib.layers.avg_pool2d(h, kernel_size=(2, 2), stride=(2, 2))

    h = tf.contrib.layers.flatten(h)

    h = tf.contrib.layers.fully_connected(h, 512, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    h = taylor(h, k=2, is_train=is_train, name="Ac/7")
    h = tf.contrib.layers.fully_connected(h, 512, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    h = taylor(h, k=2, is_train=is_train, name="Ac/8")
    h = tf.contrib.layers.fully_connected(h, n_classes)

    return h, is_train, keep_prob


def main(_):
    # Import data
    mnist = FMNIST_aug()
    mnist.data_augmentation()
    n_classes = 10

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    print("Gradient Clipping Status: "+str(clip))

    learning_rate = tf.placeholder(tf.float32)
    # Build the graph for the deep net
    y_conv, is_train, keep_prob = deepnn(x, n_classes)

    with tf.name_scope('Loss'):
        def include_activation(name):
            return ('activation_coeff' in name)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
        graph = tf.get_default_graph()
        temp = [op.values()[0] for op in graph.get_operations() if ((len(op.values()) >= 1) and (include_activation(op.values()[0].name)))]
        reg_loss = 0.01 * tf.add_n(
        [tf.reduce_sum(0.1* tf.abs(tf.cast(v, tf.float32)) + 0.0*tf.nn.l2_loss(tf.cast(v, tf.float32))) for v in temp])
        reg_loss = reg_loss 
        w_loss = tf.losses.get_regularization_loss()

    with tf.name_scope('Adam_optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if clip == True:
                optimizer = tf.train.AdamOptimizer(learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(cross_entropy+reg_loss+w_loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                # gradients = [
                    # None if gradient is None else tf.clip_by_norm(gradient, 5.0)
                    # for gradient in gradients]
                train_step = optimizer.apply_gradients(zip(gradients, variables))
            else:
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy+w_loss)

                # reg_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(w_loss)
                # train_step = tf.group(train_step, reg_step)

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
        for i in range(20000):
            batch = mnist.next_train_batch(64, augment=True)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], is_train: True, keep_prob: 0.8})
                a = []
                for batchx, batchy in iterate_minibatches(mnist.get_test_images(), mnist.get_test_labels(), 1000):
                    a.append(accuracy.eval(feed_dict={x: batchx, y_: batchy, is_train: False, keep_prob: 1.0}))
                print('Step %d, Training accuracy %g, Testing accuracy %g' % (i, train_accuracy, np.mean(a)))
                acc.append(a)

            train_step.run(feed_dict={x: batch[0], y_: batch[1], is_train: True, learning_rate: lr(i), keep_prob: 0.8})

        # print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.get_test_images(), y_: mnist.get_test_labels(), is_train: False, keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)