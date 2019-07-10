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

def step_decay(epoch):
	initial_lrate = 0.1
	if epoch <= 4:
		return 1e-3
	elif epoch < 8:
		return 1e-4
	return 1e-5

features, labels = twospirals(100000, r=1, turns=2)
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

def fit_model(bs, nb_epoch, model):
  y = Y
  optim = keras.optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
  model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['acc'])
  csv_logger = CSVLogger("logwide.csv", append=True, separator=';')
  lrate = LearningRateScheduler(step_decay)
  callbacks_list = [csv_logger, lrate]
  model.fit(scaledx_train, y_train, epochs=nb_epoch, validation_data=(scaledx_test, y_test), batch_size=bs, verbose=1, shuffle=True, callbacks=callbacks_list)

def linearnet(features):
  inputs = Input(shape=(features,))
  x = inputs
  # x = BatchNormalization()(x)
  x = Dense(1000, kernel_regularizer=regularizers.l1(0.05))(x)
  x = Dense(1000)(x)
  x = Dense(1000)(x)
  x = Dense(1, activation='sigmoid')(x)
  
  model = Model(inputs, x)
  return model

batch_size=1024
nb_epoch=100

model = linearnet(scaledx_train.shape[-1])

fit_model(batch_size, nb_epoch, model)
