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
from sklearn.linear_model import LogisticRegression

np.random.seed(0)

features, labels = twospirals(10000, r=1, turns=2)
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# plt.show()

feature_gen = PolynomialFeatures(14, include_bias=True)
X = feature_gen.fit_transform(features)
print("Number of features are {:d}".format(X.shape[-1]))

Y = labels.reshape([-1, 1])
Y = Y.astype(np.float32)

x_train, y_train, x_test, y_test = split(X, Y, split=0.8, shuffle=True)

scaler = preprocessing.StandardScaler().fit(x_train)
scaledx_train = scaler.transform(x_train)
scaledx_test = scaler.transform(x_test)

# reg = LinearRegression().fit(scaledx_train, y_train)
# reg = linear_model.Ridge(alpha=0.5).fit(scaledx_train, y_train)
clf = LogisticRegression(random_state=0, n_jobs=6, solver='saga', max_iter=1000, tol=1e-6).fit(scaledx_train, y_train)
# reg = linear_model.ElasticNet(alpha=5.0, l1_ratio=0.01).fit(scaledx_train, y_train)

print("Training Accuracy {:4f}".format(clf.score(scaledx_train, y_train)))
print("Testing Accuracy {:4f}".format(clf.score(scaledx_test, y_test)))



