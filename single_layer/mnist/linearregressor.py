import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, Input, add, concatenate
from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import keras.regularizers as regularizers
from keras.layers.normalization import BatchNormalization
from utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
import keras
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from keras.utils import np_utils

np.random.seed(0)
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[::10]
x_test = x_test[::10]
y_train = y_train[::10]
y_test = y_test[::10]

# flatten 28*28 images to a 784 vector for each image
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

feature_gen = PolynomialFeatures(2, include_bias=True)
x_train = feature_gen.fit_transform(x_train)
x_test = feature_gen.fit_transform(x_test)

print("Number of features are {:d}".format(x_train.shape[-1]))

# x_train, y_train, x_test, y_test = split(X, Y, split=0.8, shuffle=True)

scaler = preprocessing.StandardScaler().fit(x_train)
scaledx_train = scaler.transform(x_train)
scaledx_test = scaler.transform(x_test)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


reg = LinearRegression().fit(scaledx_train, y_train)
# reg = linear_model.Ridge(alpha=0.5).fit(scaledx_train, y_train)
# reg = linear_model.Lasso(alpha=0.00).fit(scaledx_train, y_train)
# reg = linear_model.ElasticNet(alpha=5.0, l1_ratio=0.01).fit(scaledx_train, y_train)

print("Training R^2 {:4f}".format(reg.score(scaledx_train, y_train)))
print("Testing R^2 {:4f}".format(reg.score(scaledx_test, y_test)))

# y_pred = reg.predict(scaledx_train)

# y_pred[y_pred>=0.5] = 1.0
# y_pred[y_pred<0.5] = 0.0

# print("Training Accuracy {:4f}".format(np.mean(1*(y_pred==y_train))))

# y_pred = reg.predict(scaledx_test)
# y_pred[y_pred>=0.5] = 1.0
# y_pred[y_pred<0.5] = 0.0

# print("Testing Accuracy {:4f}".format(np.mean(1*(y_pred==y_test))))
