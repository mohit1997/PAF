import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl

pca = PCA()

vis_tensor = np.load('vis_tensor.npy')
labels = np.argmax(np.load('labels.npy'), axis=-1)

at_path = 'conv1/Ac/1/activation_attention:0.npy'
attention = np.load(at_path)
k = attention.shape[1]

sort = np.argsort(labels)

sorted_labels = labels[sort]
sorted_vistensor = vis_tensor[sort]

print(sorted_labels)


pca_ob = pca.fit(sorted_vistensor)
print(pca_ob.explained_variance_ratio_)
final_features = pca_ob.transform(sorted_vistensor) 
print(final_features.shape)

N = len(np.unique(labels))
# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
bounds = np.linspace(0,N,N+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

sc = plt.scatter(final_features[:, 0], final_features[:, 1], c=sorted_labels, cmap=cmap, norm=norm)
plt.colorbar(sc)
plt.xlabel('PCA Feature 1 with Var_ratio ' + str(pca_ob.explained_variance_ratio_[0]))
plt.ylabel('PCA Feature 2 with Var_ratio ' + str(pca_ob.explained_variance_ratio_[1]))
plt.title("Visualizing Last Layer of CNN for Taylor k={:d}".format(k))
plt.tight_layout()
plt.savefig('vis.png')
plt.show()