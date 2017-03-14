from time import time
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA as sklearnPCA
import os,sys
import scipy.io as sio
from sklearn import cross_validation
from sklearn.model_selection import cross_val_predict,StratifiedKFold

os.chdir('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Datasets_Matched')
dataset = sio.loadmat('Autism_cl.mat')
iter = 2


if iter== 0:
	x = dataset['data_Aut']
	id = dataset['id_Aut']
	y = dataset['y_aut']
	kf_total = cross_validation.KFold(len(x), n_folds=10,shuffle=True, random_state=782828)
	a = kf_total
	sys.stdout=open('PCA_robustness_Aut.txt',"w")
elif iter==1:
	x = dataset['data_Controls']
	id = dataset['id_con']
	y = dataset['y_con']
	kf_total = cross_validation.KFold(len(x), n_folds=10,shuffle=True, random_state=782828)
	a = kf_total
	sys.stdout=open('PCA_robustness_Controls.txt',"w")
else :
	kf_total = StratifiedKFold(n_splits =10,shuffle=True)
	y = np.concatenate((dataset['y_aut'],dataset['y_con']),axis =0)
	id = np.concatenate((dataset['id_Aut'],dataset['id_con']),axis =0)
	x = np.concatenate((dataset['data_Aut'],dataset['data_Controls']),axis =0)
	kf_total.get_n_splits(x, y)
	a = kf_total.split(x,y)
	sys.stdout=open('PCA_robustness_Complete.txt',"w")

y = np.ravel(y)
mat = []
X_new = []

v = 0        
for train, test in kf_total.split(x,y):
	v = v+1 
	sklearn_pca= sklearnPCA()
	print 'try'+`v`+'\n'
	print id[train]
	print '\n'
	sklearn_trasf_pca =sklearn_pca.fit_transform(x[train])
	
	X_new.append(sklearn_pca.components_[:5,:])


mat =[]
for i in range(len(X_new)):
	for j in range(len(X_new)):
                if i!=j:
                		print [i,j]
                		mat.append(np.dot(X_new[i],X_new[j].T))

rob_value = []

for i in range(len(mat)):
	 	print [i,np.sum(mat[i].max(0))] 
	 	rob_value.append(np.sum(mat[i].max(0)))
        

plt.plot(rob_value)
plt.show()
sys.stdout.close()
