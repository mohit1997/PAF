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
training_epochs = 2000
batch_size = 265
display_step = 1
logs_path = '/tmp/tensorflow_logs/example'
decision_boundary_display = True

np.random.seed(0)
tf.set_random_seed(0)


def lr(epoch):
    if epoch < 1600:
        return 1e-3 #RAMO
    elif epoch < 3000:
        return 1e-4 #RAMO 1e-2
    elif epoch < 80:
        return 1e-3
    return 1e-4


def hidden_RAMO(is_train):
    h1 = tf.contrib.layers.fully_connected(x, 128, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05))
    h1 = taylor(h1, k=4, is_train=is_train, name='Ac/1')
    h1 = tf.contrib.layers.fully_connected(h1, 64, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05))
    h1 = taylor(h1, k=2, is_train=is_train, name='Ac/3')
    h1 = tf.keras.layers.Dense(1)(h1)
    # t = activation_function_taylor(t, k=4, is_train=is_train)
    return h1


def select_model(is_train):
    usage = 'Usage: python mnist_maxout_example.py (LINEAR|RELU|MAXOUT)'
    assert len(sys.argv) == 2, usage
    t = sys.argv[1].upper()
    print('Type = ' + t)
    if t == 'SLAF':
        return hidden_RAMO(is_train)
    elif t == 'RELU':
        return relu(is_train)
    elif t =='TAYLOR':
        return hidden_taylor(is_train)
    else:
        raise Exception('Unknown type. ' + usage)

from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

x = tf.placeholder(tf.float32, [None, x_train.shape[-1]], name='InputData')
y = tf.placeholder(tf.float32, [None, 1], name='LabelData')
learning_rate = tf.placeholder(tf.float32)

is_train = tf.placeholder(tf.bool)

pred = select_model(is_train)


t = sys.argv[1].upper()
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    def include_activation(name):
        return ('activation_coeff' in name)
    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=pred))
    # l2loss = tf.add_n(
    #       # loss is computed using fp32 for numerical stability.
    #       [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
    #        if include_activation(v.name)])
    reg_loss = tf.losses.get_regularization_loss()
    print(reg_loss)

    graph = tf.get_default_graph()
    temp = [op.values()[0] for op in graph.get_operations() if ((len(op.values()) >= 1) and (include_activation(op.values()[0].name)))]
    # w_loss = 1.0 * tf.add_n(
    # [tf.reduce_sum(0.5*tf.nn.l2_loss(tf.cast(v, tf.float32))) for v in temp])
    l1loss =  0.1 * tf.add_n([tf.reduce_sum(0.1* tf.abs(tf.cast(v, tf.float32))) for v in temp])
    loss = loss #+ w_loss
    cost = tf.sqrt(loss)
with tf.name_scope('SGD'):
    # Gradient Descent
    if t == 'SLAF':
        first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "model")
        second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "Ac")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # t1 = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=first_train_vars)
            # t2 = tf.train.AdamOptimizer(5*learning_rate).minimize(loss, var_list=second_train_vars)
            # train_op = tf.group(t1, t2)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss+l1loss)
            reg_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(reg_loss)
            train_op = tf.group(train_op, reg_step)

    else:
        first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "model")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=first_train_vars)


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
            _, c, summary, batch_acc = sess.run([train_op, reg_loss, merged_summary_op, cost], feed_dict={x: batch_x, y: batch_y, is_train: True, learning_rate: lr(epoch)})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_acc.append(batch_acc)
            avg_cost += c / total_batch
            i += 1
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:

            a = cost.eval({x: x_test, y: y_test, is_train: False})
            # prediction = logits.eval({x: mnist.test.images[0:1], y: mnist.test.labels[0:1]})
            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f},'.format(avg_cost), 'Training Average Loss :{}'.format(np.mean(avg_acc)), 'Testing Loss: {}'.format(a))

    print('Optimization Finished!')

    # Test model
    print('Accuracy:', acc.eval({x: x_test, y: y_test, is_train: False}))

