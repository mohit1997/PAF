import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x):
    return np.exp(x)/(np.sum(np.exp(x), axis=-1))

def batchnorm(x, mean, variance):
	epsilon = 1e-8
	out = (x - mean)/(np.sqrt(variance + epsilon))
	return out




at_path = 'conv1/Ac/1/activation_attention:0.npy'
ker_path = 'conv1/Ac/1/activation_weights:0.npy'
bias_path = 'conv1/Ac/1/activation_bias:0.npy'
mean_path = 'conv1/Ac/1/activation_means:0.npy'
var_path = 'conv1/Ac/1/activation_variances:0.npy'



attention = sigmoid(np.load(at_path))
k = attention.shape[1]


kernel = np.load(ker_path) + np.concatenate([np.ones([1, 1]), np.zeros([1, k-1])], axis=-1)
bias = np.load(bias_path)
mean = np.load(mean_path)
variance = np.load(var_path)

print(attention)
print(attention*kernel)

# Resize mean and variance tensor
mean = np.squeeze(mean)
variance = np.squeeze(variance)


x = np.linspace(-1, 1, 1000).reshape(-1, 1)
l = []
for i in range(k):
    temp = x**(i+1)
    l.append(temp)

inp = np.concatenate(l, axis=1)

inp = np.expand_dims(inp, axis=1)

# Normalize Input with saved mean and variance
inp = batchnorm(inp, mean, variance)

output = np.sum(kernel*attention*inp, axis=-1) + bias

C = output.shape[1]

rows = np.ceil(np.sqrt(C))

fig = plt.figure(figsize=(12, 12))

for i in range(C):
    ax = fig.add_subplot(rows, rows, i+1)
    ax.plot(x, output[:, i])
    ax.set_title('Channel ' + str(i+1))
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.grid()

fig.suptitle('Activation Plots for Taylor with k={:d}'.format(k))
fig.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig("Activations_k={:d}.png".format(k))
plt.show()

