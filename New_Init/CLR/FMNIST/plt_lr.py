import numpy as np
import matplotlib.pyplot as plt

loss = np.load('list_checklr.npy')
loss = loss[:450]

index = np.arange(len(loss))

def lr(iteration, low=1e-3, high=1.0, steps=1000):
    current = low + (high-low)/steps*iteration
    return current

lrs = np.log10(lr(index, steps=2000))

plt.plot(lrs, loss)
plt.xlabel('Learning Rate')
plt.ylabel('Log Loss')

plt.title('LR Test')

plt.savefig('LRtest.png')
plt.show()