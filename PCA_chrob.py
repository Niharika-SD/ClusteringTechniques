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
kf_total = StratifiedKFold(n_splits =50,shuffle=True)
kf_total.get_n_splits(x, y)
 
X_new = []        
for train, test in kf_total.split(x,y):

	sklearn_pca= sklearnPCA()
	print x[train].shape
	sklearn_trasf_pca =sklearn_pca.fit_transform(x[train])
	
	X_new.append(sklearn_pca.components_[:5,:])

#sys.stdout=open('PCA_robustness.txt',"w")
mat =[]
for i in range(len(X_new)):
	for j in range(len(X_new)):
                if i!=j:
                        mat.append(np.dot(X_new[i],X_new[j].T))

rob_value = []

for i in range(len(mat)):
        rob_value.append(np.sum(mat[i].max(0)))

plt.plot(rob_value)
plt.show()
#sys.stdout.close()
