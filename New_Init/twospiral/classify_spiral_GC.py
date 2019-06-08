import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from RAMO_fast import RAMO
from approximate_taylor_fast import activation_function_taylor
from taylor_norm import taylor
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# learning_rate = 5e-2
training_epochs = 15
batch_size = 10
display_step = 1
logs_path = '/tmp/tensorflow_logs/example'
decision_boundary_display = True

np.random.seed(0)
tf.set_random_seed(0)

def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)


def twospirals(n_points, r=100, turns=1, noise=.1):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    n = np.sort(n, axis=0)
    print(n)
    d1x = -np.cos(n*turns)*n*r + np.random.rand(n_points,1) * noise
    d1y = np.sin(n*turns)*n*r + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))

X, labels = twospirals(10000, r=1, turns=2)
Y = labels.reshape([-1, 1])



def split(X, Y, split=0.5, shuffle=False):
    l = len(X)

    if shuffle:
        indices = np.arange(l)
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

    tr = int(split * l)

    x_train = X[:tr]
    y_train = Y[:tr]
    x_test = X[tr:]
    y_test = Y[tr:]

    return x_train, y_train, x_test, y_test


def split_cluster(X, Y):
    l = int(len(X)/2)
    x_train1, y_train1, x_test1, y_test1 = split(X[:l], Y[:l])
    x_train2, y_train2, x_test2, y_test2 = split(X[l:], Y[l:])

    x_train, y_train = np.concatenate([x_train1, x_train2], axis=0), np.concatenate([y_train1, y_train2], axis=0)    
    x_test, y_test = np.concatenate([x_test1, x_test2], axis=0), np.concatenate([y_test1, y_test2], axis=0)    

    return x_train, y_train, x_test, y_test  

enc = OneHotEncoder()
Y = Y.astype(np.float32)

X1, Y1, x_test, y_test = split_cluster(X, Y)

x_train, y_train, x_test, y_test = split(X1, Y1, split=0.8, shuffle=True)

Y1 = enc.fit_transform(Y.reshape(-1, 1)).toarray()
X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
plt.figure()

plt.scatter(x_train[:, 0], x_train[:, 1], c=np.float32(2*y_train[:, 0] + 5)/15)
plt.scatter(x_test[:, 0], x_test[:, 1], c=np.float32(2*y_test[:, 0] + 15)/15)

plt.show()

sys.exit()
###
# Y = enc.fit_transform(Y.reshape(-1, 1)).toarray()

def lr(epoch):
    if epoch < 6:
        return 1e-2 #RAMO
    elif epoch < 15:
        return 1e-3 #RAMO 1e-2
    elif epoch < 20:
        return 1e-3
    return 1e-4


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        # if(start_idx + batchsize >= inputs.shape[0]):
        #   break;

        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


x = tf.placeholder(tf.float32, [None, 2], name='InputData')
y = tf.placeholder(tf.float32, [None, 1], name='LabelData')
learning_rate = tf.placeholder(tf.float32)

def relu(is_train):
    n_classes = 1
    dim = 2
    val = 10
    with tf.variable_scope("model"):
        W1 = create_weight_variable('Weights', [dim, val])
        b1 = create_bias_variable('Bias', [val])

        W2 = create_weight_variable('Weights2', [val, val])
        b2 = create_bias_variable('Bias2', [val])

        W3 = create_weight_variable('Weights3', [val, val])
        b3 = create_bias_variable('Bias3', [val])

        W4 = create_weight_variable('Weights4', [val, n_classes])
        b4 = create_bias_variable('Bias4', [n_classes])
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    h3 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    t = (tf.matmul(h3, W4) + b4)
    # t = activation_function_taylor(t, k=4, is_train=is_train)
    return t, t

def hidden_taylor(is_train):
    n_classes = 1
    dim = 2
    val = 10
    with tf.variable_scope("model"):
        W1 = create_weight_variable('Weights', [dim, val])
        b1 = create_bias_variable('Bias', [val])

        W2 = create_weight_variable('Weights2', [val, val])
        b2 = create_bias_variable('Bias2', [val])

        W3 = create_weight_variable('Weights3', [val, val])
        b3 = create_bias_variable('Bias3', [val])

        W4 = create_weight_variable('Weights4', [val, n_classes])
        b4 = create_bias_variable('Bias4', [n_classes])
    # h1 = taylor(tf.matmul(x, W1) + b1, k=2)
    # h3 = taylor(tf.matmul(h1, W2) + b2, k=2)
    t = (tf.matmul(h3, W4) + b4)
    # t = activation_function_taylor(t, k=4, is_train=is_train)
    return t, t

def hidden_RAMO(is_train):
    n_classes = 1
    dim = 2
    with tf.variable_scope("model"):
        val = 10
        W1 = create_weight_variable('Weights', [dim, val])
        b1 = create_bias_variable('Bias', [val])

        W2 = create_weight_variable('Weights2', [val, val])
        b2 = create_bias_variable('Bias2', [val])

        # W3 = create_weight_variable('Weights3', [5, 5])
        # b3 = create_bias_variable('Bias3', [5])

        W4 = create_weight_variable('Weights4', [val, n_classes])
        b4 = create_bias_variable('Bias4', [n_classes])
    h1 = taylor(tf.matmul(x, W1) + b1, k=4, is_train=is_train, name='Ac/1')
    h3 = taylor(tf.matmul(h1, W2) + b2, k=4, is_train=is_train, name='Ac/2')
    # h1 = RAMO(tf.matmul(x, W1) + b1, k1=3, k2=3, is_train=True, name='Ac/1')
    # h3 = RAMO(tf.matmul(h1, W2) + b2, k1=3, k2=3, is_train=True, name='Ac/2')
    # h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    # h3 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    # h1 = activation_function_taylor(tf.matmul(x, W1) + b1, k=4)
    # h3 = activation_function_taylor(tf.matmul(h1, W2) + b2, k=4)

    t = (tf.matmul(h3, W4) + b4)
    # t = activation_function_taylor(t, k=4, is_train=is_train)
    return t, t


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


is_train = tf.placeholder(tf.bool)

pred, logits = select_model(is_train)


t = sys.argv[1].upper()
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    def include_activation(name):
        return ('activation_weights' in name)
    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred))
    # l2loss = tf.add_n(
    #       # loss is computed using fp32 for numerical stability.
    #       [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
    #        if include_activation(v.name)])
    loss = loss #+ l2loss
    cost = loss
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
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            train_op = optimizer.apply_gradients(zip(gradients, variables))

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
        x_min, x_max = X1[:, 0].min() - .5, X1[:, 0].max() + .5
        y_min, y_max = X1[:, 1].min() - .5, X1[:, 1].max() + .5
        h = 0.1
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        Z = sess.run(values, feed_dict={x: (np.c_[xx.ravel(), yy.ravel()]), is_train: True})
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X1[:, 0], X1[:, 1], c=Y1.reshape(-1), cmap=plt.cm.Spectral)
        plt.savefig('twp_spiral.png')

