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

def update_op(pop_max, max_tensor, m, n, decay):
    update_max = tf.assign(pop_max, max_tensor*(1-decay) + pop_max*(decay))
    with tf.control_dependencies([update_max]):
        max_tensor = m*max_tensor + n
        return max_tensor


def taylor(inp, k, is_train, name="activation", scale=None, decay=0.99):
    # input_r = tf.reshape(inp, [-1, 1])
    # shape = inp.get_shape().as_list()
    # rank = len(shape)
    # input_r = inp
    # l = []
    # for i in range(k):
    #     temp = tf.pow(input_r, i+1)
    #     # print(temp)
    #     l.append(temp)

    # X = tf.stack(l, axis=(rank))

    X = inp

    # with tf.variable_scope("activation", reuse=tf.AUTO_REUSE):
    #     input_r = tf.layers.batch_normalization(input_r, axis=-1, training=is_train)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # name = random_id(10)
        m = tf.Variable(2, dtype=tf.float32, name=random_id(10))
        n = tf.abs(tf.Variable(1e-8, dtype=tf.float32, name="n", trainable=False))
        print(n)

        # Xin = X
        
        # for i in range(k1+k2):
        #     if i == 0:
        #         w = create_variables(random_id(10), [1], scale=scale)*0 + 1e-8
        #         c = np.float32(np.pi) * (i/2.0)
        #         temp = tf.multiply(w, tf.cos(c* X))
        #     else:
        #         if i%2 == 0:
        #             w = create_variables(random_id(10), [1], scale=scale)/(k1+k2)
        #             c = np.float32(np.pi) * (i/2.0)
        #             temp += tf.multiply(w, tf.cos(c* X))
        #         else:
        #             w = create_variables(random_id(10), [1], scale=scale)/(k1+k2)
        #             c = np.float32(np.pi) * (i/2.0)
        #             temp += tf.multiply(w, tf.sin(c* X))

        # return temp

        # print("SHAPE IS", len(Xin.get_shape()))


        # Xin = X/max_val
        if len(X.get_shape()) == 4:


            (batch, H, W, C) = X.get_shape()
            # m = tf.Variable(2*tf.ones([1, 1, 1, C]), dtype=tf.float32, name="m")
            Xreshape = X
            # Xreshape = tf.reshape(X, [-1, H*W*C, 1, 1])
            max_tensor = tf.reduce_max(Xreshape, axis=[0, 1, 2], keep_dims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=0, keep_dims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=1, keep_dims=True)
            min_tensor = tf.reduce_min(Xreshape, axis=[0, 1, 2], keep_dims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=0, keep_dims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=1, keep_dims=True)
            max_tensor = tf.abs(tf.maximum(max_tensor, -min_tensor))

            pop_max = tf.Variable(tf.ones([1, 1, 1, C]), trainable = False)
            max_tensor = tf.cond(is_train, lambda: update_op(pop_max, max_tensor, m, n, decay), lambda: m*pop_max + n)
            Xin = X/max_tensor

            shape = Xin.get_shape().as_list()
            rank = len(shape)
            input_r = Xin
            l = []
            for i in range(k):
                temp = tf.pow(input_r, i+1)
                # print(temp)
                l.append(temp)

            Xin = tf.stack(l, axis=(rank))

            # Xin = tf.expand_dims(Xin, -1)
            # x = np.pi/2*np.arange(1, 2*k1, 2)
            # x.resize((1, 1, 1, k1))
            # constant1 = tf.convert_to_tensor(x, np.float32)
            # # print(constant1.get_shape())


            # y = np.pi*np.arange(1,k2+1)
            # y.resize((1, 1, 1, k2))
            # constant2 = tf.convert_to_tensor(y, np.float32)
            # # print(constant2.get_shape())


            # out1 = tf.sin(tf.multiply(constant1, Xin))
            # out2 = tf.cos(tf.multiply(constant2, Xin))

            # print(out1.get_shape())
            # out = tf.concat([out1, out2], axis=3)

            # kernel = tf.Variable(np.random.randn(1, 1, k1+k2-1, 1), dtype=tf.float32)
            kernel = create_variables("activation_weights", [k], scale=scale)
            # kernel = tf.Variable(tf.random_normal([k1+k2]))
            # conv = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding='SAME')

            # # print(conv.get_shape())
            # conv = tf.squeeze(conv, [-1])
            # return(conv)
            bias = bias_variable([1])
            conv = tf.reduce_sum(kernel * Xin, axis=-1) + bias
            return(conv)



        if len(X.get_shape()) == 3:
            (batch, W, C) = X.get_shape()
            # m = tf.Variable(2*tf.ones([1, 1, C]), dtype=tf.float32, name=random_id(10))
            Xreshape = X
            # Xreshape = tf.reshape(X, [-1, W*C, 1])
            max_tensor = tf.reduce_max(Xreshape, axis=[0, 1], keep_dims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=0, keep_dims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=1, keep_dims=True)
            min_tensor = tf.reduce_min(Xreshape, axis=[0, 1], keep_dims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=0, keep_dims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=1, keep_dims=True)
            max_tensor = tf.abs(tf.maximum(max_tensor, -min_tensor))

            pop_max = tf.Variable(tf.ones([1, 1, C]), trainable = False)
            max_tensor = tf.cond(is_train, lambda: update_op(pop_max, max_tensor, m, n, decay), lambda: m*pop_max + n)
            Xin = X/max_tensor
            
            shape = Xin.get_shape().as_list()
            rank = len(shape)
            input_r = Xin
            l = []
            for i in range(k):
                temp = tf.pow(input_r, i+1)
                # print(temp)
                l.append(temp)

            Xin = tf.stack(l, axis=(rank))
            # Xin = tf.expand_dims(Xin, -1)
            # x = np.pi/2*np.arange(1, 2*k1, 2)
            # x.resize((1, 1, 1, k1))
            # constant1 = tf.convert_to_tensor(x, np.float32)
            # # print(constant1.get_shape())


            # y = np.pi*np.arange(1,k2+1)
            # y.resize((1, 1, 1, k2))
            # constant2 = tf.convert_to_tensor(y, np.float32)
            # # print(constant2.get_shape())


            # out1 = tf.sin(tf.multiply(constant1, Xin))
            # out2 = tf.cos(tf.multiply(constant2, Xin))

            # print(out1.get_shape())
            # out = tf.concat([out1, out2], axis=3)

            # kernel = tf.Variable(np.random.randn(1, 1, k1+k2-1, 1), dtype=tf.float32)
            kernel = create_variables("activation_weights", [k], scale=scale)
            # kernel = tf.Variable(tf.random_normal([k1+k2]))
            # conv = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding='SAME')

            # # print(conv.get_shape())
            # conv = tf.squeeze(conv, [-1])
            # return(conv)
            bias = bias_variable([1])
            conv = tf.reduce_sum(kernel * Xin, axis=-1) + bias
            return(conv)

        if len(X.get_shape()) == 2:
            (batch, N) = X.get_shape()
            # m = tf.Variable(2*tf.ones([1, N]), dtype=tf.float32, name=random_id(10))
            Xreshape = X
            # Xreshape = tf.reshape(X, [-1, N])
            max_tensor = tf.reduce_max(Xreshape, axis=[0], keep_dims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=0, keep_dims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=1, keep_dims=True)
            min_tensor = tf.reduce_min(Xreshape, axis=[0], keep_dims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=0, keep_dims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=1, keep_dims=True)
            max_tensor = tf.abs(tf.maximum(max_tensor, -min_tensor))

            pop_max = tf.Variable(tf.ones([1, N]), trainable = False)

            max_tensor = tf.cond(is_train, lambda: update_op(pop_max, max_tensor, m, n, decay), lambda: m*pop_max + n)
            Xin = X/max_tensor
            # Xin = tf.expand_dims(Xin, -1)
            # x = np.pi/2*np.arange(1, 2*k1, 2)
            # x.resize((1, 1, 1, k1))
            # constant1 = tf.convert_to_tensor(x, np.float32)
            # # print(constant1.get_shape())


            # y = np.pi*np.arange(1,k2+1)
            # y.resize((1, 1, 1, k2))
            # constant2 = tf.convert_to_tensor(y, np.float32)
            # # print(constant2.get_shape())


            # out1 = tf.sin(tf.multiply(constant1, Xin))
            # out2 = tf.cos(tf.multiply(constant2, Xin))

            # print(out1.get_shape())
            # out = tf.concat([out1, out2], axis=3)

            # kernel = tf.Variable(np.random.randn(1, 1, k1+k2-1, 1), dtype=tf.float32)
            shape = Xin.get_shape().as_list()
            rank = len(shape)
            input_r = Xin
            l = []
            for i in range(k):
                temp = tf.pow(input_r, i+1)
                # print(temp)
                l.append(temp)

            Xin = tf.stack(l, axis=(rank))
            kernel = create_variables("activation_weights", [k], scale=scale)
            # kernel = tf.Variable(tf.random_normal([k1+k2]))
            # conv = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding='SAME')

            # # print(conv.get_shape())
            # conv = tf.squeeze(conv, [-1])
            # return(conv)
            bias = bias_variable([1])
            conv = tf.reduce_sum(kernel * Xin, axis=-1) + bias
            return(conv)


    # size = 1
    # for i in range(rank-1):
    #     size = size*shape[i+1]
    # input_temp = tf.reshape(input_r, [-1, size*(k+1)])
    # print(input_temp)
    # _, _, out = add_layer1(input_temp, k+1)

    return input_r
