import numpy as np
import tensorflow as tf
from time import time
import math
import os


from include.data import get_data_set
from include.model import model, lr

def loss_filter_fn(name):
    return ("activation_coeff" in name)


train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")

tf.set_random_seed(0)
np.random.seed(0)   

x, y, output, y_pred_cls, global_step, learning_rate, is_train = model()

graph = tf.get_default_graph()
temp = [op.values()[0] for op in graph.get_operations() if loss_filter_fn(op.values()[0].name)]

tf.set_random_seed(0)
np.random.seed(0)   
     
global_accuracy = 0
epoch_start = 0

# PARAMS
_BATCH_SIZE = 128
_EPOCH = 20
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"
weight_decay = 0.05


# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
l2_loss = weight_decay * tf.add_n(
    [tf.reduce_sum(tf.abs(tf.cast(v, tf.float32))) for v in temp])

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step)
reg_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(l2_loss)

optimizer = tf.group(optimizer, reg_optimizer)

# PREDICTION AND ACCURACY CALCULATION
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# SAVER
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def train(epoch):
    global epoch_start
    epoch_start = time()
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    i_global = 0
    
    indices = np.arange(len(train_x))
    np.random.shuffle(indices)

    print(indices)

    for s in range(batch_size):
        batch_xs = train_x[indices[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]]
        batch_ys = train_y[indices[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]]

        start_time = time()
        i_global, _, batch_loss, batch_acc = sess.run(
            [global_step, optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch), is_train: True})
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))

            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))

    test_and_save(i_global, epoch)


def test_and_save(_global_step, epoch):
    global global_accuracy
    global epoch_start

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)

    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch), is_train: False}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format((epoch+1), acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))

    file = open("accuracy.txt", "a+")
    file.write(mes.format((epoch+1), acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))
    file.close()

    if global_accuracy != 0 and global_accuracy < acc:

        # summary = tf.Summary(value=[
        #     tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        # ])
        # train_writer.add_summary(summary, _global_step)

        # saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy)) 

        global_accuracy = acc

        vis_tensor = np.zeros(shape=(test_y.shape[0], test_y.shape[1]))

        i = 0

        while i < len(test_x):
            j = min(i + _BATCH_SIZE, len(test_x))
            batch_xs = test_x[i:j, :]
            batch_ys = test_y[i:j, :]
            vis_tensor[i:j] = sess.run(
                output,
                feed_dict={x: batch_xs, y: batch_ys, is_train: False}
            )
            i = j
        np.save('vis_tensor', vis_tensor)
        np.save('labels', test_y)

        for v in tf.global_variables():
          if ("activation_" in v.name):
            print(v)
            var = sess.run(v)
            directory = os.path.dirname(str(v.name))
            if not os.path.exists(directory):
              os.makedirs(directory)
            np.save(str(v.name), var)

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")


def main():
    train_start = time()
    file = open("accuracy.txt", "w+")  
    file.close() 
    for i in range(_EPOCH):
        print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
        train(i)

    hours, rem = divmod(time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format(global_accuracy, int(hours), int(minutes), seconds))


if __name__ == "__main__":
    main()


sess.close()
