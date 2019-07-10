import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from taylor_batch_norm import taylor
import sys
import os
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# learning_rate = 5e-2
training_epochs = 15
batch_size = 256
display_step = 1
logs_path = '/tmp/tensorflow_logs/example'
decision_boundary_display = True

np.random.seed(0)
tf.set_random_seed(0)


def lr(epoch):
    if epoch < 6:
        return 1e-3 #RAMO
    elif epoch < 10:
        return 1e-3 #RAMO 1e-2
    elif epoch < 20:
        return 1e-4
    return 1e-4


def relu(is_train):
    n_classes = 1
    val = 40
    h1 = tf.contrib.layers.fully_connected(x, val)
    h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=is_train))
    h1 = tf.contrib.layers.fully_connected(h1, val)
    h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=is_train))
    t = tf.contrib.layers.fully_connected(h1, n_classes)
    return t


def select_model(is_train):
    usage = 'Usage: python mnist_maxout_example.py (LINEAR|RELU|MAXOUT)'
    assert len(sys.argv) == 2, usage
    t = sys.argv[1].upper()
    print('Type = ' + t)
    if t == 'SLAF':
        return hidden_RAMO(is_train)
    elif t == 'RELU':
        return relu(is_train)
    else:
        raise Exception('Unknown type. ' + usage)

X, labels = twospirals(100000, r=1, turns=2)
Y = labels.reshape([-1, 1])

Y = Y.astype(np.float32)

# x_train, y_train, x_test, y_test = split_cluster(X, Y)

x_train, y_train, x_test, y_test = split(X, Y, split=0.8, shuffle=True)


x = tf.placeholder(tf.float32, [None, 2], name='InputData')
y = tf.placeholder(tf.float32, [None, 1], name='LabelData')
learning_rate = tf.placeholder(tf.float32)

is_train = tf.placeholder(tf.bool)

pred = select_model(is_train)


t = sys.argv[1].upper()
with tf.name_scope('Loss'):

    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred))
    loss = loss #+ l2loss
    cost = loss
with tf.name_scope('SGD'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.name_scope('Accuracy'):
    predictions = tf.nn.sigmoid(pred)
    values = tf.round(predictions)
    correct_pred = tf.equal(values, y)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Accuracy
    # p_class = tf.argmax(pred, 1)
    # t_class = tf.argmax(y, 1)
    # acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Create a summary to monitor cost tensor
tf.summary.scalar('loss', cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar('accuracy', acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(x_train) / batch_size)
        avg_acc = []
        # Loop over all batches
        i = 0
        print(lr(epoch))
        for batch_x, batch_y in iterate_minibatches(x_train, y_train, batchsize=batch_size, shuffle=True):
            _, c, summary, batch_acc = sess.run([train_op, cost, merged_summary_op, acc], feed_dict={x: batch_x, y: batch_y, is_train: True, learning_rate: lr(epoch)})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_acc.append(batch_acc)
            avg_cost += c / total_batch
            i += 1
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:

            a = acc.eval({x: x_test, y: y_test, is_train: False})
            # prediction = logits.eval({x: mnist.test.images[0:1], y: mnist.test.labels[0:1]})
            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f},'.format(avg_cost), 'trainavg_acc :{}'.format(np.mean(avg_acc)), 'Validation Accuracy: {}'.format(a))

    print('Optimization Finished!')

    # Test model
    # Calculate accuracy
    print('Accuracy:', acc.eval({x: x_test, y: y_test, is_train: False}))


    if decision_boundary_display:
        x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
        y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
        h = 0.1
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        Z = sess.run(values, feed_dict={x: (np.c_[xx.ravel(), yy.ravel()]), is_train: False})
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.reshape(-1), cmap=plt.cm.Spectral)
        plt.xlabel("Feature 1")
        plt.xlabel("Feature 2")
        plt.tight_layout()
        plt.savefig('twospiralrelu.png')

