import numpy as np
import tensorflow as tf
import sys
import os
from sklearn.preprocessing import OneHotEncoder

def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)



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
    x_train1, y_train1, x_test1, y_test1 = split(X[:l], Y[:l], split=0.8, shuffle=True)
    x_train2, y_train2, x_test2, y_test2 = split(X[l:], Y[l:], split=0.8, shuffle=True)

    x_train, y_train = np.concatenate([x_train1, x_train2], axis=0), np.concatenate([y_train1, y_train2], axis=0)    
    x_test, y_test = np.concatenate([x_test1, x_test2], axis=0), np.concatenate([y_test1, y_test2], axis=0)    

    return x_train, y_train, x_test, y_test  

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