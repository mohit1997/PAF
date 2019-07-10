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

# features, labels = twospirals(100000, r=1, turns=2)
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# plt.show()

feature_gen = PolynomialFeatures(8, include_bias=True)
x_train = feature_gen.fit_transform(x_train)
x_test = feature_gen.fit_transform(x_test)

print("Number of features are {:d}".format(x_train.shape[-1]))

# x_train, y_train, x_test, y_test = split(X, Y, split=0.8, shuffle=True)

scaler = preprocessing.StandardScaler().fit(x_train)
scaledx_train = scaler.transform(x_train)
scaledx_test = scaler.transform(x_test)

# reg = LinearRegression().fit(scaledx_train, y_train)
# reg = linear_model.Ridge(alpha=0.5).fit(scaledx_train, y_train)
reg = linear_model.Lasso(alpha=0.01).fit(scaledx_train, y_train)
# reg = linear_model.ElasticNet(alpha=5.0, l1_ratio=0.01).fit(scaledx_train, y_train)

print("Training R^2 {:4f}".format(reg.score(scaledx_train, y_train)))
print("Testing R^2 {:4f}".format(reg.score(scaledx_test, y_test)))

y_pred = reg.predict(scaledx_train)
print("Training RMSE {:4f}".format(np.sqrt(np.mean((y_pred - y_train)**2))))
y_pred = reg.predict(scaledx_test)
print("Testing RMSE {:4f}".format(np.sqrt(np.mean((y_pred - y_test)**2))))


