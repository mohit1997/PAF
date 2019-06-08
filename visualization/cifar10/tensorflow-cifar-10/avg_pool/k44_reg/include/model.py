import tensorflow as tf
from include.taylor_batch_norm import taylor

def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        is_train = tf.placeholder(tf.bool)
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:
        conv = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[3, 3],
            padding='SAME',
            activation=None
        )
        conv = taylor(conv, k=4, is_train=is_train, name="Ac/1")
        # conv = tf.nn.relu(conv)
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME',
            activation=None
        )
        conv = taylor(conv, k=4, is_train=is_train, name="Ac/2")
        # conv = tf.nn.relu(conv)
        # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        pool = tf.layers.average_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        # drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)
        drop = pool

    with tf.variable_scope('conv2') as scope:
        conv = tf.layers.conv2d(
            inputs=drop,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=None
        )
        # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        pool = tf.layers.average_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=128,
            kernel_size=[2, 2],
            padding='SAME',
            activation=None
        )
        # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        pool = tf.layers.average_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        # drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)
        drop = pool

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(drop, [-1, 4 * 4 * 128])

        fc = tf.layers.dense(inputs=flat, units=1500, activation=None)
        # drop = tf.layers.dropout(fc, rate=0.5)
        drop = fc
        logits = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, activation=None, name=scope.name)

    y_pred_cls = tf.argmax(logits, axis=1)

    return x, y, logits, y_pred_cls, global_step, learning_rate, is_train


# Taylor
def lr(epoch):
    if epoch < 10:
        return 5e-3
    if epoch < 15:
        return 5e-4
    return 1e-5

# RelU
def lr(epoch):
    if epoch < 10:
        return 1e-3
    return 1e-4
