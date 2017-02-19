from time import time
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import os,sys,glob

os.chdir('/home/ndsouza4/matlab/New_files/Correlation_data/classify/rz/')
dataset = sio.loadmat('Aut2cl.mat')
X= dataset['data']
y = dataset['y']
y = np.ravel(y)
n_samples, n_features = X.shape
n_neighbors = 7
sklearn_kpca = sklearnKPCA(n_components=3,kernel="poly", degree =3)
sklearn_transf_kpca = sklearn_kpca.fit_transform(X)


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig= plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1],X[i,2], color=plt.cm.Set1(y[i] / 10.))
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=3, init=sklearn_kpca.fit_transform(X), random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the data kPCA initialisation(time %.2fs)" %
               (time() - t0))
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the data PCA initialisation(time %.2fs)" %
               (time() - t0))

plt.show()
