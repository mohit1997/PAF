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
  initial = tf.constant(0.1, shape=shape)
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


def apq(inp, is_train=False, name="activation", scale=None, decay=0.99):

    X = inp

    with tf.variable_scope(name, reuse=False):
        # name = random_id(10)

        if len(X.get_shape()) == 4:


            (batch, H, W, C) = X.get_shape()

        #     shape = X.get_shape().as_list()
        #     rank = len(shape)
        #     input_r = X
        #     l = []
        #     for i in range(k):
        #         temp = tf.pow(input_r, i+1)
        #         # print(temp)
        #         l.append(temp)
        #     X = tf.stack(l, axis=(rank))

        #     batch_mean, batch_var = tf.nn.moments(X, axes=[0, 1, 2], keep_dims=True)
        #     pop_mean = tf.Variable(tf.zeros([1, 1, 1, C, k]), trainable = False)
        #     pop_var = tf.Variable(tf.ones([1, 1, 1, C, k]), trainable = False)
        #     # max_tensor = tf.cond(is_train, lambda: update_op(pop_max, max_tensor, m, n, decay), lambda: m*max_tensor + n)
        #     Xin = tf.cond(is_train, lambda: update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay), lambda: tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8))
            
            bias = create_variables("bias", [C])
            b = create_variables("b", [C])
            c = create_variables("c", [C])
            a1 = create_variables("a1/activation_weights", [C])
            a2 = create_variables("a2/activation_weights", [C])
            # a3 = create_variables("a3/activation_weights", [C])
            a3 = 0

            out = a1*tf.math.maximum(0.0, b-X) + a2*tf.math.maximum(0.0, -X) + tf.math.maximum(0.0, X) + a3*tf.math.maximum(0.0, (X-b)*(-X))

            return out



        if len(X.get_shape()) == 3:
            (batch, W, C) = X.get_shape()

        #     shape = X.get_shape().as_list()
        #     rank = len(shape)
        #     input_r = X
        #     l = []
        #     for i in range(k):
        #         temp = tf.pow(input_r, i+1)
        #         # print(temp)
        #         l.append(temp)
        #     X = tf.stack(l, axis=(rank))

        #     batch_mean, batch_var = tf.nn.moments(X, axes=[0, 1], keep_dims=True)
        #     pop_mean = tf.Variable(tf.zeros([1, 1, C, k]), trainable = False)
        #     pop_var = tf.Variable(tf.ones([1, 1, C, k]), trainable = False)
        #     # max_tensor = tf.cond(is_train, lambda: update_op(pop_max, max_tensor, m, n, decay), lambda: m*max_tensor + n)
        #     Xin = tf.cond(is_train, lambda: update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay), lambda: tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8))

            bias = create_variables("bias", [C])
            b = create_variables("b", [C])
            c = create_variables("c", [C])
            a1 = create_variables("a1/activation_weights", [C])
            a2 = create_variables("a2/activation_weights", [C])
            # a3 = create_variables("a3/activation_weights", [C])
            a3 = 0

            out = a1*tf.math.maximum(0.0, b-X) + a2*tf.math.maximum(0.0, -X) + tf.math.maximum(0.0, X) + a3*tf.math.maximum(0.0, (X-b)*(-X))

            return out

        if len(X.get_shape()) == 2:
            (batch, N) = X.get_shape()

            # shape = X.get_shape().as_list()
            # rank = len(shape)
            # input_r = X
            # l = []
            # for i in range(k):
            #     temp = tf.pow(input_r, i+1)
            #     # print(temp)
            #     l.append(temp)
            # X = tf.stack(l, axis=(rank))

            # batch_mean, batch_var = tf.nn.moments(X, axes=[0], keep_dims=True)
            # pop_mean = tf.Variable(tf.zeros([1, N]), trainable = False)
            # pop_var = tf.Variable(tf.ones([1, N]), trainable = False)
            # # # max_tensor = tf.cond(is_train, lambda: update_op(pop_max, max_tensor, m, n, decay), lambda: m*max_tensor + n)
            # Xin = tf.cond(is_train, lambda: update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay), lambda: tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8))

            # kernel = create_variables("activation_weights", [N, k], scale=scale)
            
            # bias = bias_variable(name="bias", shape=[N])
            # conv = tf.reduce_sum(kernel * Xin, axis=-1) + bias
            bias = create_variables("bias", [N])
            b = create_variables("b", [N])
            c = create_variables("c", [N])
            a1 = create_variables("a1/activation_weights", [N])
            a2 = create_variables("a2/activation_weights", [N])
            # a3 = create_variables("a3/activation_weights", [N])
            a3 = 0

            out = a1*tf.math.maximum(0.0, b-X) + a2*tf.math.maximum(0.0, -X) + tf.math.maximum(0.0, X) + a3*tf.math.maximum(0.0, (X-b)*(-X))

            return out