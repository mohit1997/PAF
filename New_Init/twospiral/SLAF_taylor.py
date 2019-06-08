from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_variables(name, shape, initializer=tf.initializers.random_normal(), scale=None):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = None
    if scale is not None:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.05)

    new_variables = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def bias_variable(name, shape, trainable=True):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.0, shape=shape)
  return tf.get_variable(name=name, dtype=tf.float32, trainable=trainable, initializer=initial,
                                    regularizer=None)


def random_id(length):
    number = '0123456789'
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    n = len(number)
    m = len(alpha)
    id = ''
    for i in range(0,length,2):

        id += number[np.random.choice(n)]
        id += alpha[np.random.choice(m)]
    return id

def update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay):
    update_mean = tf.assign(pop_mean, batch_mean*(1-decay) + pop_mean*(decay))
    update_var = tf.assign(pop_var, batch_var*(1-decay) + pop_var*(decay))
    with tf.control_dependencies([update_mean, update_var]):
        temp = tf.nn.batch_normalization(X, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=1e-8)
        return temp


def taylor(inp, k, is_train=False, name="activation", scale=None, decay=0.99):

    X = inp

    with tf.variable_scope(name, reuse=False):
        # name = random_id(10)

        m = bias_variable(name='m', shape=[1])*20.0
        # n = tf.abs(tf.Variable(1, dtype=tf.float32, name=random_id(10)))
        n = tf.abs(bias_variable(name='n', trainable=True, shape=[1])*1e-7)
        print(n)

        if len(X.get_shape()) == 4:


            (batch, H, W, C) = X.get_shape()

            shape = X.get_shape().as_list()
            rank = len(shape)
            input_r = X
            l = []
            for i in range(k):
                temp = tf.pow(input_r, i+1)
                # print(temp)
                l.append(temp)
            X = tf.stack(l, axis=(rank))

            batch_mean, batch_var = tf.nn.moments(X, axes=[0, 1, 2], keep_dims=True)
            pop_mean = tf.Variable(tf.zeros([1, 1, 1, C, k]), trainable = False)
            pop_var = tf.Variable(tf.ones([1, 1, 1, C, k]), trainable = False)
            if type(is_train) is not bool:
                Xin = tf.cond(is_train, lambda: update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay), lambda: tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8))
            else:
                if is_train:
                    Xin = update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay)
                else:
                    Xin = tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8)
            
            kernel = create_variables("activation_weights", [C, k], scale=scale)

            bias = bias_variable(name="activation_bias", shape=[C])
            conv = tf.reduce_sum(kernel * Xin, axis=-1) + bias
            return(conv)



        if len(X.get_shape()) == 3:
            (batch, W, C) = X.get_shape()

            shape = X.get_shape().as_list()
            rank = len(shape)
            input_r = X
            l = []
            for i in range(k):
                temp = tf.pow(input_r, i+1)
                # print(temp)
                l.append(temp)
            X = tf.stack(l, axis=(rank))

            batch_mean, batch_var = tf.nn.moments(X, axes=[0, 1], keep_dims=True)
            pop_mean = tf.Variable(tf.zeros([1, 1, C, k]), trainable = False)
            pop_var = tf.Variable(tf.ones([1, 1, C, k]), trainable = False)
            if type(is_train) is not bool:
                Xin = tf.cond(is_train, lambda: update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay), lambda: tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8))
            else:
                if is_train:
                    Xin = update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay)
                else:
                    Xin = tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8)

            kernel = create_variables("activation_weights", [C, k], scale=scale)
            
            bias = bias_variable(name="activation_bias", shape=[C])
            conv = tf.reduce_sum(kernel * Xin, axis=-1) + bias
            return(conv)

        if len(X.get_shape()) == 2:
            (batch, N) = X.get_shape()

            shape = X.get_shape().as_list()
            rank = len(shape)
            input_r = X
            l = []
            for i in range(k):
                temp = tf.pow(input_r, i+1)
                # print(temp)
                l.append(temp)
            X = tf.stack(l, axis=(rank))

            batch_mean, batch_var = tf.nn.moments(X, axes=[0], keep_dims=True)
            pop_mean = tf.Variable(tf.zeros([1, N, k]), trainable = False)
            pop_var = tf.Variable(tf.ones([1, N, k]), trainable = False)
            if type(is_train) is not bool:
                Xin = tf.cond(is_train, lambda: update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay), lambda: tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8))
            else:
                if is_train:
                    Xin = update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay)
                else:
                    Xin = tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8)

            kernel = create_variables("activation_weights", [N, k], scale=scale)
            
            bias = bias_variable(name="activation_bias", shape=[N])
            conv = tf.reduce_sum(kernel * Xin, axis=-1) + bias
            return(conv)