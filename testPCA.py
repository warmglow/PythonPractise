import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)
X = np.dot(np.random.rand(2, 2), np.random.rand(2, 2000)).T

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(X_pca, np.zeros_like(X_pca), alpha=0.5, color='red')
plt.title('PCA Result')
plt.show() # this is a reversion

