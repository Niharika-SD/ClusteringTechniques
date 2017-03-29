from sklearn import metrics
from sklearn.cluster import KMeans,SpectralClustering
import scipy.io as sio
import os,sys
import numpy as np
from matplotlib import pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA

os.chdir('/home/niharika-shimona/Documents/')
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

data = sio.loadmat('/home/niharika-shimona/Documents/Projects/Autism_Network/code/dis_affinity_IP_e.mat')
x = data['dis_affinity_IP']

pca = PCA(n_components = 10)
x = pca.fit_transform(x)

y = sio.loadmat('/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data_time_course/y_label.mat')
data_1 = sio.loadmat('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Comparative_Affinity/dis_affinity116.mat')
n_labels = len(np.unique(y['y']))
labels = np.ravel(y['y'])
distance = data_1['dis_affinity']

delta = 5
beta =0.09
t0 = time.time()
dist_matrix = np.exp(-beta * distance / distance.std())
X = np.exp(- dist_matrix ** 2 / (2. * delta ** 2))
estimator = SpectralClustering(n_clusters=2, eigen_solver=None, random_state=None, n_init=10, 
	               gamma=0.01, affinity='precomputed', assign_labels='kmeans')

y_pred = estimator.fit_predict(X)
t1 = time.time()

print("Spectral Clustering run \n")
print(79 * '_')
print('% 9s' % 'init' '    time   homo   compl  v-meas     ARI AMI  silhouette')
print('% 9s  %.4fs  %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % ('E frob dis', (t1 - t0),
             metrics.homogeneity_score(labels, estimator.labels_), 
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(X, estimator.labels_,
                                      metric='euclidean')))
y_pred = estimator.labels_.astype(np.int)
fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:,2], color=colors[y_pred+1].tolist(), s=10)
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.xticks(())
# plt.yticks(())
# plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15,horizontalalignment='right')
plt.show()