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

np.random.seed(0)
decision_boundary_display = True

features, labels = twospirals(20000, r=1, turns=2)
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# plt.show()

feature_gen = PolynomialFeatures(25, include_bias=True)
X = feature_gen.fit_transform(features)
print("Number of features are {:d}".format(X.shape[-1]))

Y = labels.reshape([-1, 1])
Y = Y.astype(np.float32)

x_train, y_train, x_test, y_test = split(X, Y, split=0.8, shuffle=True)

print("Number of features are {:d}".format(x_train.shape[-1]))

# x_train, y_train, x_test, y_test = split(X, Y, split=0.8, shuffle=True)

scaler = preprocessing.StandardScaler().fit(x_train)
scaledx_train = scaler.transform(x_train)
scaledx_test = scaler.transform(x_test)

reg = LinearRegression().fit(scaledx_train, y_train)
# reg = linear_model.Ridge(alpha=0.5).fit(scaledx_train, y_train)
# reg = linear_model.Lasso(alpha=0.00).fit(scaledx_train, y_train)
# reg = linear_model.ElasticNet(alpha=5.0, l1_ratio=0.01).fit(scaledx_train, y_train)

print("Training R^2 {:4f}".format(reg.score(scaledx_train, y_train)))
print("Testing R^2 {:4f}".format(reg.score(scaledx_test, y_test)))

y_pred = reg.predict(scaledx_train)

y_pred[y_pred>=0.5] = 1.0
y_pred[y_pred<0.5] = 0.0

print("Training Accuracy {:4f}".format(np.mean(1*(y_pred==y_train))))

y_pred = reg.predict(scaledx_test)
y_pred[y_pred>=0.5] = 1.0
y_pred[y_pred<0.5] = 0.0

print("Testing Accuracy {:4f}".format(np.mean(1*(y_pred==y_test))))


if decision_boundary_display:
    x_min, x_max = features[:, 0].min() - .5, features[:, 0].max() + .5
    y_min, y_max = features[:, 1].min() - .5, features[:, 1].max() + .5
    h = 0.1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    inputs = np.c_[xx.ravel(), yy.ravel()]
    inputs = feature_gen.fit_transform(inputs)
    inputs = scaler.transform(inputs)
    print(inputs.shape)
    Z = reg.predict(inputs)
    Z = Z.reshape(xx.shape)
    Z[Z>=0.5] = 1.0
    Z[Z<0.5] = 0.0
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    # plt.scatter(features[:, 0], features[:, 1], c=Y.reshape(-1), cmap=plt.cm.Spectral)
    plt.savefig('twp_spiral.png')
    plt.show()
