from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from taylor_batch_norm import taylor

import tensorflow as tf
np.random.seed(0)
tf.set_random_seed(0)


FLAGS = None
def include_activation(name):
    return ('activation_attention' in name)

def lr(iteration):
  if iteration < 3000:
    return 1e-2
  elif iteration < 5000:
    return 1e-3
  elif iteration < 15000:
    1e-4
  return 5e-5


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Relu_W_file = np.genfromtxt('Relu_W.csv', delimiter=',')
# Relu_B_file = np.genfromtxt('Relu_B.csv', delimiter=',')

# Relu_W = Relu_W_file#.astype(dtype = 'float64')#[:,0]
# Relu_B = Relu_B_file#.astype(dtype = 'float64')#[:,0]
# print("-----------------")
# print(Relu_W.shape)
# print("-----------------")

def deepnn(x, is_train):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])
    temp_1 = b_conv1
    #a = tf.reshape(tf.size(x_image), [-1, 1])
    temp_1 = (conv2d(x_image, W_conv1) + b_conv1) #UC
    # _, _,out,_, _ = non_linear_Ac(temp_1, input_size = 25088, k1 = 4, k2 = 4, output_size = 25088) #UC
    # h_conv1, w1, b1 = RAMO(temp_1, k1=4, k2=4, scale=None) #UC
    h_conv1 = taylor(temp_1, k=4, is_train=is_train, name="Ac/1")
    # h_conv1 = tf.nn.relu(temp_1)

    print(h_conv1, "RAJANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #C

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 16 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])
    temp_c2 = (conv2d(h_pool1, W_conv2) + b_conv2) #UC
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  #C
    # _,_,out_c2,_, _ = non_linear_Ac(temp_c2, input_size = 14*14*32, k1 = 4, k2 = 4, output_size = 14*14*32) #UC
    # h_conv2 = tf.reshape(out_c2, [-1, 14, 14, 32]) #UC
    h_conv2 = taylor(temp_c2, k=4, is_train=is_train, name="Ac/2")
    # h_conv2 = tf.nn.relu(temp_c2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x32 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 32, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32]) 
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #C
    h_fc = tf.matmul(h_pool2_flat, W_fc1) + b_fc1 #UC
    # w, b, h_fc1, _, m = non_linear_Ac(h_fc, input_size=1024, k1=8, k2=8, scale=None, output_size=1024) #UC
    h_fc1 = taylor(h_fc, k=5, is_train=is_train, name="Ac/3")
    # h_fc1 = tf.nn.relu(h_fc)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  is_train = tf.placeholder(tf.bool)

  learning_rate = tf.placeholder(tf.float32)

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x, is_train=is_train)

  with tf.name_scope('loss'):
    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # l2_loss = .005 * tf.add_n(
    #   # loss is computed using fp32 for numerical stability.
    #   [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
    #    if include_activation(v.name)])
    print([v.name for v in tf.trainable_variables() if include_activation(v.name)])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    t1 = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # t2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss)
    # train_step = tf.group(t1, t2)
    train_step = t1

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())
  acc = []
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(25000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0, is_train: True})
        print('step %d, training accuracy %g' % (i, train_accuracy))


        a1 = accuracy.eval(feed_dict={
        x: mnist.test.images[0:5000, :], y_: mnist.test.labels[0:5000, :], keep_prob: 1.0, is_train: False})
        a2 = accuracy.eval(feed_dict={
            x: mnist.test.images[5000:, :], y_: mnist.test.labels[5000:, :], keep_prob: 1.0, is_train: False})
        print(a1/2 + a2/2)
        acc.append(a1/2 + a2/2)

      # print(batch[0][0])
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, learning_rate: lr(i), is_train: True})

    a1 = accuracy.eval(feed_dict={
        x: mnist.test.images[0:5000, :], y_: mnist.test.labels[0:5000, :], keep_prob: 1.0, is_train: False})
    a2 = accuracy.eval(feed_dict={
        x: mnist.test.images[5000:, :], y_: mnist.test.labels[5000:, :], keep_prob: 1.0, is_train: False})
    print(a1/2 + a2/2)

    np.save('testing_score_taylor_batch_norm', np.array(acc))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
