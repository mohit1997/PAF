import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA()

vis_tensor = np.load('vis_tensor.npy')
labels = np.argmax(np.load('labels.npy'), axis=-1)

sort = np.argsort(labels)

sorted_labels = labels[sort]
sorted_vistensor = vis_tensor[sort]

print(sorted_labels)


pca_ob = pca.fit(sorted_vistensor)
final_features = pca_ob.transform(sorted_vistensor) 
print(final_features.shape)

plt.scatter(final_features[:, 0], final_features[:, 1], c=sorted_labels, cmap='Accent')
plt.show()