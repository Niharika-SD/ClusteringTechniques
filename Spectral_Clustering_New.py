from sklearn import metrics
from sklearn.cluster import KMeans,SpectralClustering
import scipy.io as sio
import os,sys
import numpy as np
from matplotlib import pyplot as plt
import time

os.chdir('/home/niharika-shimona/Documents/')
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

data = sio.loadmat('patient_data_red_rz.mat')
data_1 = sio.loadmat('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Comparative_Affinity/dis_affinity116.mat')
dist_matrix = data_1['dis_affinity']

delta = 0.01
t0 = time.time()
X = np.exp(- dist_matrix ** 2 / (2. * delta ** 2))
spectral = SpectralClustering(n_clusters=2, eigen_solver=None, random_state=None, n_init=10, 
	               gamma=1.0, affinity='precomputed', assign_labels='kmeans')

y = spectral.fit_predict(X)
t1 = time.time()

print spectral.labels_
print metrics.silhouette_score(X, spectral.labels_,
                                      metric='euclidean')
y_pred = spectral.labels_.astype(np.int)
plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred+1].tolist(), s=10)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xticks(())
plt.yticks(())
plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
plt.show()