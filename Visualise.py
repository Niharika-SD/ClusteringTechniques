import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from pylab import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import KernelPCA as sklearnKPCA
import scipy.io as sio
import os

os.chdir('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Datasets_Matched')

dataset = sio.loadmat('Autism_cl.mat')

y = np.concatenate((dataset['y_aut'],dataset['y_con']),axis =0)
id = np.concatenate((dataset['id_Aut'],dataset['id_con']),axis =0)
x = np.concatenate((dataset['data_Aut'],dataset['data_Controls']),axis =0)

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig= plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1],X[i,2], color=plt.cm.Set1(y[i] +1 / 10.))
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

sklearn_pca = sklearnPCA(n_components=30)
sklearn_transf = sklearn_pca.fit(x)


plot_embedding(sklearn_transf.transform(x),"PCA visualisation")
plt.show()

sklearn_kpca = sklearnKPCA(n_components=5,kernel="rbf")
sklearn_transf_kpca = sklearn_kpca.fit(x)

plot_embedding(sklearn_transf_kpca.transform(x),"kPCA visualisation")
plt.show()



