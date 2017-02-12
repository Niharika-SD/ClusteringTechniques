import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from pylab import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import KernelPCA as sklearnKPCA
import scipy.io as sio
import os

os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('Aut_classify2rz.mat')
x= dataset['data']
y = dataset['y']
y = np.ravel(y)

sklearn_pca = sklearnPCA(n_components=30)
sklearn_transf = sklearn_pca.fit_transform(x)
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')

for i in xrange(y.shape[0]):
	if y[i] ==0:
		lo = ax.scatter(sklearn_transf[i,0],sklearn_transf[i,1],sklearn_transf[i,2],  c='r', marker='o')
	elif y[i] ==1:
	    ll = ax.scatter(sklearn_transf[i,0],sklearn_transf[i,1],sklearn_transf[i,2],  c='g', marker='^')
	else:
	    lm = ax.scatter(sklearn_transf[i,0],sklearn_transf[i,1],sklearn_transf[i,2],  c='b', marker='*')

ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
ax.set_zlabel('dim 3')
plt.legend((lo,ll,lm),('controls','ASP','HFA'))
plt.title('PCA visualisation based on first 3 components \n')


sklearn_kpca = sklearnKPCA(n_components=5,kernel="rbf")
sklearn_transf_kpca = sklearn_kpca.fit_transform(x)
ax = fig.add_subplot(212, projection='3d')

for i in xrange(y.shape[0]):
	if y[i] ==0:
		li = ax.scatter(sklearn_transf_kpca[i,0],sklearn_transf_kpca[i,1],sklearn_transf_kpca[i,2],  c='r', marker='o',)
	elif y[i] == 1:
	    lv = ax.scatter(sklearn_transf_kpca[i,0],sklearn_transf_kpca[i,1],sklearn_transf_kpca[i,2],  c='g', marker='^')
	else:
	    lw = ax.scatter(sklearn_transf_kpca[i,0],sklearn_transf_kpca[i,1],sklearn_transf_kpca[i,2],  c='b', marker='*')

ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
ax.set_zlabel('dim 3')
plt.legend((li,lv,lw),('controls','ASP','HFA'))
plt.title('kPCA visualisation based on first 3 components \n')
plt.show()




