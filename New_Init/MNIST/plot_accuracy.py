import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

score_emfn = np.load('testing_score_emfn.npy')
score_relu = np.load('testing_score_relu.npy')
score_taylor_max = np.load('testing_score_taylor_max.npy')
score_taylor_gc = np.load('testing_score_taylor_gradient_clipping.npy')
score_taylor_bn = np.load('testing_score_taylor_batch_norm.npy')

alpha = 50
emfn = score_emfn[alpha:]
relu = score_relu[alpha:]
taylor_max = score_taylor_max[alpha:]
taylor_gc = score_taylor_gc[alpha:]
taylor_bn = score_taylor_bn[alpha:]

emfn_max = np.max(emfn)
relu_max = np.max(relu)
taylor_max_max = np.max(taylor_max)
taylor_gc_max = np.max(taylor_gc)
taylor_bn_max = np.max(taylor_bn)

emfn_max_index = np.argmax(emfn)
relu_max_index = np.argmax(relu)
taylor_max_max_index = np.argmax(taylor_max)
taylor_gc_max_index = np.argmax(taylor_gc)
taylor_bn_max_index = np.argmax(taylor_bn)

print(emfn_max_index, relu_max_index, taylor_max_max_index, taylor_gc_max_index, taylor_bn_max_index)

x_vals = 50 + np.arange(len(taylor_bn))
# plt.plot(emfn, 'r', label=('EMFN (max= {:.2f}%)'.format(emfn_max*100)), linestyle="--")
plt.plot(x_vals, relu, 'b', linestyle='-', label=('ReLU ({:.2f}%)'.format(relu_max*100)))
# plt.plot(taylor_max, 'g', label=('Mc Poly_MS (max= {:.2f}%)'.format(taylor_max_max*100)))
plt.plot(x_vals, taylor_bn, 'k', linestyle='--', label=('SLAF ({:.2f}%)'.format(taylor_bn_max*100)))
# plt.plot(taylor_gc, 'c', label=('Mc Poly_GC (max= {:.2f}%)'.format(taylor_gc_max*100)))

plt.legend()
plt.grid()
plt.title('Comparison on MNIST Dataset')
plt.ylabel('Testing Accuracy')
plt.xlabel('Iterations (x100)')
plt.tight_layout()
plt.savefig('AccuracyCurvefinal.pdf')