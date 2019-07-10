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

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=random_id(10))



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


def activation_function_taylor(inp, k, is_train=False, scale=None):
    # input_r = tf.reshape(inp, [-1, 1])
    shape = inp.get_shape().as_list()
    rank = len(shape)
    input_r = inp
    l = []
    for i in range(k):
        temp = tf.pow(input_r, i+1)
        # print(temp)
        l.append(temp)

    input_r = tf.stack(l, axis=(rank))

    # with tf.variable_scope("activation", reuse=tf.AUTO_REUSE):
    #     input_r = tf.layers.batch_normalization(input_r, axis=-1, training=is_train)

    with tf.variable_scope("activation", reuse=tf.AUTO_REUSE):
        Weights = create_variables('weights', [k], scale)
        Bias = bias_variable([1])


    input_r = tf.multiply(Weights, input_r)
    # print("After Multiplication", input_r.get_shape().as_list())
    input_r = tf.reduce_sum(input_r, axis=-1) + Bias

    print(input_r)
    # size = 1
    # for i in range(rank-1):
    #     size = size*shape[i+1]
    # input_temp = tf.reshape(input_r, [-1, size*(k+1)])
    # print(input_temp)
    # _, _, out = add_layer1(input_temp, k+1)

    return input_r
