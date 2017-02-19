from time import time
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA as sklearnPCA
import os,sys
import scipy.io as sio
from sklearn.model_selection import cross_val_predict,StratifiedKFold

os.chdir('/home/ndsouza4/matlab/New_files/Correlation_data/classify/rz/')

dataset = sio.loadmat('Aut2cl.mat')

x = dataset['data']
y = dataset['y']
y = np.ravel(y)
mat = []
kf_total = StratifiedKFold(n_splits =20,shuffle=True)
kf_total.get_n_splits(x, y)
ct =1 

for train, test in kf_total.split(x,y):

	if ct >1:
		X_prev = X_new

	sklearn_pca= sklearnPCA()
	print x[train].shape
	sklearn_trasf_pca =sklearn_pca.fit_transform(x[train])
	
	X_new = sklearn_pca.components_[:5,:]
	print(X_new.shape)

	if ct >1:
		mat.append(np.dot(X_prev,X_new.T))

	ct = ct+1

print mat[9]

sys.stdout=open('PCA_robustness.txt',"w")

for i in range(len(mat)):
	print ('\n')
	print(mat[i])

sys.stdout.close()



