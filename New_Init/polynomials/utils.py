import numpy as np

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

def get_polynomial(num_features=10, num_terms=None, degree=3, samples=10000):
	X = np.random.normal(0, 1, (samples, num_features))
	if num_terms is None:
		num_terms=num_features
	# ind = [np.random.randint(num_features, size=i) for i in range(1, degree+1)]
	ind = [np.random.randint(num_features, size=i) for i in degree*np.ones(num_terms, dtype=np.int64)]
	print(ind)
	Y = 0
	for places in ind:
		Y = Y + np.prod(X[:, places], axis=1)

	return X, Y.reshape(-1, 1)

def split(X, Y, train_split=0.5, shuffle=False):
    l = len(X)

    if shuffle:
        indices = np.arange(l)
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

    tr = int(train_split * l)

    x_train = X[:tr]
    y_train = Y[:tr]
    x_test = X[tr:]
    y_test = Y[tr:]

    return x_train, y_train, x_test, y_test