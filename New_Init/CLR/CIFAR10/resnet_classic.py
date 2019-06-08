import tensorflow as tf
from utils import *
import numpy as np
from imgaug import augmenters as iaa
import tempfile

np.random.seed(0)
tf.set_random_seed(0)

max_iter = 20000

def lr(iteration, low=0.08, high=0.8, steps=max_iter, midstep=6000):
    assert midstep < steps/2

    min_lr = 1e-7
    endstep = steps - 2*midstep

    if iteration <= midstep:
        lr = low + (high-low)/midstep*iteration

    if iteration > midstep and iteration < 2*midstep:
        lr = high - (high-low)/(midstep)*(iteration-midstep)

    if iteration >= 2*midstep:
        lr = low - (low - min_lr)/(endstep)*(iteration - 2*midstep)

    return lr

def mom(iteration, low=0.85, high=0.95, steps=max_iter, midstep=6000):
    assert midstep < steps/2

    endstep = steps - 2*midstep

    if iteration <= midstep:
        lr = high - (high-low)/midstep*iteration

    if iteration > midstep and iteration < 2*midstep:
        lr = low + (high-low)/(midstep)*(iteration-midstep)

    if iteration >= 2*midstep:
        lr = high

    return lr

def resnet_module(data, K, stride, is_train, red=False):
	# the true data
	shortcut = h = data
	# h = tf.layers.batch_normalization(h, training=is_train)
	# h = tf.nn.relu(h)
	# h = tf.layers.conv2d(h, filters=int(K*0.25), kernel_size=(1, 1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.005))
	h = tf.layers.conv2d(h, filters=int(K), kernel_size=(3, 3), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.005))
	h = tf.layers.batch_normalization(h, training=is_train)
	h = tf.nn.relu(h)

	h = tf.layers.conv2d(h, filters=int(K), kernel_size=(3, 3), strides=stride, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.005))
	h = tf.layers.batch_normalization(h, training=is_train)
	

	if red:
		shortcut = tf.layers.conv2d(shortcut, filters=int(K), kernel_size=(1, 1), strides=stride, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.005))

	h = h + shortcut

	out = tf.nn.relu(h)

	return out


def resnet(x, is_train, n_classes, stages=(2, 2, 2, 2), filters=(64, 64, 128, 256, 512)):
	orig = h = x

	h = tf.layers.conv2d(h, filters=int(filters[0]), kernel_size=(7, 7), strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.005))
	h = tf.layers.batch_normalization(h, training=is_train)
	h = tf.nn.relu(h)

	h = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(h)

	for i in range(len(stages)):

		stride = (1, 1) if i == 0 else (2, 2)

		h = resnet_module(h, filters[i+1], stride, is_train, red=True) if i!=0 else resnet_module(h, filters[i+1], stride, is_train, red=False)

		for j in range(stages[i]-1):
			h = resnet_module(h, filters[i+1], (1, 1), is_train, red=False)

	h = tf.layers.batch_normalization(h, training=is_train)
	h = tf.nn.relu(h)
	print(h)
	# h = tf.keras.layers.AveragePooling2D((3, 3))(h)
	h = tf.keras.layers.GlobalAveragePooling2D()(h)

	h = tf.contrib.layers.flatten(h)
	h = tf.contrib.layers.fully_connected(h, n_classes, activation_fn=None)

	return h

def main():
	n_classes = 10
	cifar = CIFAR10()
	cifar.data_augmentation()

	x = tf.placeholder(tf.float32, [None, 32, 32, 3])
	y_ = tf.placeholder(tf.float32, [None, n_classes])

	learning_rate = tf.placeholder(tf.float32)
	momentum = tf.placeholder(tf.float32)

	weight_decay = 5e-2

	is_train = tf.placeholder(tf.bool)

	out = resnet(x, is_train, n_classes)

	with tf.name_scope('Loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
															logits=out)
		cross_entropy = tf.reduce_mean(cross_entropy)
		reg_loss = tf.losses.get_regularization_loss()

	with tf.name_scope('Adam_optimizer'):
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			# train_step = tf.contrib.opt.AdamWOptimizer(0.1*learning_rate, learning_rate).minimize(cross_entropy)
			# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
			train_step = tf.train.MomentumOptimizer(learning_rate, momentum=momentum).minimize(cross_entropy)
			reg_step = tf.train.GradientDescentOptimizer(weight_decay).minimize(reg_loss)

			train_step = tf.group(train_step, reg_step)

	with tf.name_scope('Accuracy'):
		correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
		accuracy = tf.reduce_mean(correct_prediction)

	graph_location = tempfile.mkdtemp()
	print('Saving graph to: %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())
	acc = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(max_iter):
			batch = cifar.next_train_batch(512, augment=True)
			if i % 200 == 0:
				train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], is_train: True})
				a = accuracy.eval(feed_dict={x: cifar.get_test_images(), y_: cifar.get_test_labels(), is_train: False})
				print('Step %d, Training accuracy %g, Testing accuracy %g' % (i, train_accuracy, a))
				acc.append(a)

			train_step.run(feed_dict={x: batch[0], y_: batch[1], is_train: True, learning_rate: lr(i, steps=max_iter), momentum: mom(i, steps=max_iter)})

		print('test accuracy %g' % accuracy.eval(feed_dict={x: cifar.get_test_images(), y_: cifar.get_test_labels(), is_train: False}))


if __name__ == "__main__":
	main()








