import numpy as np
import tensorflow as tf
import time



def create_variables(name, shape, initializer=tf.initializers.random_normal(), is_fc_layer=False, scale=None):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if scale is not None:
        regularizer = tf.contrib.layers.l2_regularizer(scale=scale)
    else:
        regularizer = None

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

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv1d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def RAMO(X, k1=1, k2=1, scale=None, name="activation", is_train=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        name = random_id(10)
        # max_1 = tf.reduce_max(X, axis=1, keep_dims=True)
        # min_1 = tf.reduce_min(X, axis=1, keep_dims=True)
        # max_val = tf.abs(tf.maximum(-min_1, max_1))
        # # mean, var = tf.nn.moments(X, axes=[0], keep_dims=True)


        # sh = X.get_shape().as_list()
        # shp = [1]*len(sh)
        # shp = tuple(shp)
        # init_val = 5*np.ones(shp)

        # m = tf.Variable(2, dtype=tf.float32)
        # # n = tf.Variable(1, dtype=tf.float32)
        # # m = tf.maximum(m, 1.1)
        # # m = 5

        # max_val = max_val*m

        # Xin = (X)/max_val

        m = tf.Variable(2, dtype=tf.float32, name=random_id(10))
        n = tf.abs(tf.Variable(1, dtype=tf.float32, name=random_id(10)))
        

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
            Xreshape = tf.reshape(X, [-1, H*W*C, 1, 1])
            max_tensor = tf.reduce_max(Xreshape, axis=1, keepdims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=1, keepdims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=1, keepdims=True)
            min_tensor = tf.reduce_min(Xreshape, axis=1, keepdims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=1, keepdims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=1, keepdims=True)
            max_tensor = tf.abs(tf.maximum(max_tensor, -min_tensor))
            max_tensor = max_tensor*m + n
            Xin = X/max_tensor
            
            Xin = tf.expand_dims(Xin, -1)
            x = np.pi/2*np.arange(1, 2*k1, 2)
            x.resize((1, 1, 1, 1, k1))
            constant1 = tf.convert_to_tensor(x, np.float32)
            # print(constant1.get_shape())


            y = np.pi*np.arange(1,k2+1)
            y.resize((1, 1, 1, 1, k2))
            constant2 = tf.convert_to_tensor(y, np.float32)
            # print(constant2.get_shape())


            out1 = tf.sin(tf.multiply(constant1, Xin))
            out2 = tf.cos(tf.multiply(constant2, Xin))

            # print(out1.get_shape())
            out = tf.concat([out1, out2], axis=4)

            # kernel = tf.Variable(np.random.randn(1, 1, 1, k1+k2, 1), dtype=tf.float32)
            kernel = create_variables(name, [k1+k2], scale=scale)
            # kernel = tf.Variable(tf.random_normal([k1+k2]))
            bias = bias_variable([1])
            # conv = tf.nn.conv3d(out, kernel, [1, 1, 1, 1, 1], padding='SAME')

            # print(conv.get_shape())
            # conv = tf.squeeze(conv, [-1])
            conv = tf.reduce_sum(kernel * out, axis=-1) + bias
            # print(conv.get_shape())
            # conv = tf.squeeze(conv, [-1])
            return(conv)

        if len(X.get_shape()) == 3:
            (batch, H, W) = X.get_shape()
            Xreshape = tf.reshape(X, [-1, H*W, 1])
            max_tensor = tf.reduce_max(Xreshape, axis=1, keepdims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=1, keepdims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=1, keepdims=True)
            min_tensor = tf.reduce_min(Xreshape, axis=1, keepdims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=1, keepdims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=1, keepdims=True)
            max_tensor = tf.abs(tf.maximum(max_tensor, -min_tensor))
            max_tensor = max_tensor*m + n
            Xin = X/max_tensor
            Xin = tf.expand_dims(Xin, -1)
            x = np.pi/2*np.arange(1, 2*k1, 2)
            x.resize((1, 1, 1, k1))
            constant1 = tf.convert_to_tensor(x, np.float32)
            # print(constant1.get_shape())


            y = np.pi*np.arange(1,k2+1)
            y.resize((1, 1, 1, k2))
            constant2 = tf.convert_to_tensor(y, np.float32)
            # print(constant2.get_shape())


            out1 = tf.sin(tf.multiply(constant1, Xin))
            out2 = tf.cos(tf.multiply(constant2, Xin))

            # print(out1.get_shape())
            out = tf.concat([out1, out2], axis=3)

            # kernel = tf.Variable(np.random.randn(1, 1, k1+k2-1, 1), dtype=tf.float32)
            kernel = create_variables(name, [k1+k2], scale=scale)
            # kernel = tf.Variable(tf.random_normal([k1+k2]))
            # conv = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding='SAME')

            # # print(conv.get_shape())
            # conv = tf.squeeze(conv, [-1])
            # return(conv)
            bias = bias_variable([1])
            conv = tf.reduce_sum(kernel * out, axis=-1) + bias
            return(conv)

        if len(X.get_shape()) == 2:
            (batch, N) = X.get_shape()
            Xreshape = tf.reshape(X, [-1, N])
            max_tensor = tf.reduce_max(Xreshape, axis=1, keepdims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=1, keepdims=True)
            # max_tensor = tf.reduce_max(max_tensor, axis=1, keepdims=True)
            min_tensor = tf.reduce_min(Xreshape, axis=1, keepdims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=1, keepdims=True)
            # min_tensor = tf.reduce_min(min_tensor, axis=1, keepdims=True)
            max_tensor = tf.abs(tf.maximum(max_tensor, -min_tensor))
            max_tensor = max_tensor*m + n
            Xin = X/max_tensor
            Xin = tf.expand_dims(Xin, -1)
            x = np.pi/2*np.arange(1, 2*k1, 2)
            x.resize((1, 1, k1))
            constant1 = tf.convert_to_tensor(x, np.float32)
            # print(constant1.get_shape())


            y = np.pi*np.arange(1,k2+1)
            y.resize((1, 1, k2))
            constant2 = tf.convert_to_tensor(y, np.float32)
            # print(constant2.get_shape())


            out1 = tf.sin(tf.multiply(constant1, Xin))
            out2 = tf.cos(tf.multiply(constant2, Xin))

            # print(out1.get_shape())
            out = tf.concat([out1, out2], axis=2)

            # kernel = tf.Variable(np.random.randn(1, k1+k2-1, 1), dtype=tf.float32)
            # kernel = tf.Variable(tf.random_normal([k1+k2]))
            kernel = create_variables(name, [k1+k2], scale=scale)
            # conv = tf.nn.conv1d(out, kernel, stride=1, padding='SAME')
            bias = bias_variable([1])
            conv = tf.reduce_sum(kernel * out, axis=-1) + bias
            # print(conv.get_shape())
            # conv = tf.squeeze(conv, [-1])
            return(conv)