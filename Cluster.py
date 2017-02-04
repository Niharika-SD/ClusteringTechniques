import numpy as np
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pylab import *
from sklearn.decomposition import KernelPCA 
import scipy.io as sio
import os

os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('Aut_classify1rz.mat')
data= dataset['data']
y = dataset['y']
y = np.ravel(y)

np.random.seed(42)

n_samples, n_features = data.shape
n_labels = len(np.unique(y))
labels = y
sample_size = 30

print("n_labels: %d, \t n_samples %d, \t n_features %d"
      % (n_labels, n_samples, n_features))


print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_labels, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_labels, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = KernelPCA(n_components=10,kernel ='poly').fit(data)
bench_k_means(KMeans(init='k-means++', n_clusters=n_labels, n_init=1),
              name="PCA-based",
              data=data)
print(79 * '_')


reduced_data = KernelPCA(n_components=10,kernel ='rbf').fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_labels, n_init=10)
kmeans.fit(reduced_data)

y_pred = kmeans.predict(reduced_data)
count = 0
for i in xrange(y.shape[0]):
	print y_pred[i],y[i]
	if y[i] == y_pred[i]:
		count = count+1
print 100*count/y.shape[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in xrange(y.shape[0]):
	if y[i] == 0:
		ax.scatter(reduced_data[i, 0], reduced_data[i, 1],reduced_data[i, 2], '^',c ='r')
	else:	
		ax.scatter(reduced_data[i, 0], reduced_data[i, 1],reduced_data[i, 2], 'o',c ='g')
		# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1],centroids[:, 2],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.show()