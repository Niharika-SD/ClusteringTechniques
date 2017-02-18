import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
import sys,glob
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn import metrics
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.decomposition import PCA
from pylab import *
from sklearn.decomposition import KernelPCA 
import scipy.io as sio
import os

#module for metric based comparison for kmeans
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    
    print('% 9s   %.4fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
                                      # sample_size=sample_size)))
def bench_spectral(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    
    print('% 9s  %.4fs  %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_), 
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# module for comparing the performance of different clustering runs 
# based on full dataset and the reduced dataset
def k_means_performance_comparison(data,n_labels):

	bench_k_means(KMeans(init='k-means++', n_clusters=n_labels, n_init=10),
		name="k-means++", data=data)
	bench_k_means(KMeans(init='random', n_clusters=n_labels, n_init=1),
		name="random", data=data)
	## in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
	pca = PCA(n_components=10).fit(data)
	kpca =KernelPCA(n_components=5,kernel = 'rbf')
	bench_k_means(KMeans(init='k-means++',n_clusters=n_labels, n_init=10),
              name="PCA-based",data=pca.fit_transform(data))
	bench_k_means(KMeans(init='k-means++',n_clusters=n_labels, n_init=10),
              name="kPCA-based",data= kpca.fit_transform(data))

def Spectral_performance_comparison(data,n_labels):
	bench_spectral(SpectralClustering(n_clusters=n_labels,affinity="nearest_neighbors",n_neighbors = 20,assign_labels ='kmeans'),
		name="SC NN", data=data)
	bench_spectral(SpectralClustering(n_clusters=n_labels,affinity="rbf",assign_labels ='kmeans'),
		name="SC rbf", data=data)
	pca = PCA(n_components=30).fit(data)
	kpca =KernelPCA(n_components=5,kernel = 'rbf')
	bench_spectral(SpectralClustering(n_clusters=n_labels,affinity="nearest_neighbors",n_neighbors = n_labels,assign_labels ='kmeans'),
		name="SC NN pca", data=pca.fit_transform(data))
	bench_spectral(SpectralClustering(n_clusters=n_labels,affinity="nearest_neighbors",n_neighbors = n_labels,assign_labels ='kmeans'),
		name="SC NN kpca", data=kpca.fit_transform(data))

def k_means_visualise(estimator,data,n_labels,y):
	reduced_data = estimator.fit_transform(data)
	kmeans = KMeans(init='k-means++', n_clusters=n_labels, n_init=10)
	kmeans.fit(reduced_data)
	y_pred = kmeans.predict(reduced_data)
	count = 0
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in xrange(y.shape[0]):
		if y[i] == 0:
			li = ax.scatter(reduced_data[i, 0], reduced_data[i, 1],reduced_data[i, 2], '^',c ='r')
		elif y[i] ==1:	
			ll = ax.scatter(reduced_data[i, 0], reduced_data[i, 1],reduced_data[i, 2], 'o',c ='g')
		else: 
			le =ax.scatter(reduced_data[i, 0], reduced_data[i, 1],reduced_data[i, 2], 'o',c ='b')
    
	centroids = kmeans.cluster_centers_
	ax.scatter(centroids[:, 0], centroids[:, 1],centroids[:, 2],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)
	if n_labels ==3:
		plt.legend((li,ll,le),('C1','C2','C3'))
	else:
		plt.legend((li,ll),('C1','C2'))
        return fig

if __name__ == "__main__" :

	os.chdir('/home/ndsouza4/matlab/New_files/Correlation_data/classify/wrz/')
        for filename in glob.glob('A*.mat'):
            dataset = sio.loadmat(filename)
            tfilename = filename.split('.')
            sys.stdout=open(tfilename[0]+'.txt',"w")
            print filename
            
            data= dataset['data']
            y = dataset['y']
            y = np.ravel(y)
            np.random.seed(42)
            n_samples, n_features = data.shape
            n_labels = len(np.unique(y))
            labels = y
            print("K means run \n")
            print("n_labels: %d, \t n_samples %d, \t n_features %d"
                  % (n_labels, n_samples, n_features))
            print(79 * '_')
            print('% 9s' % 'init'
                  '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
            k_means_performance_comparison(data,n_labels)
            print(79 * '_')

            print("Spectral Clustering run \n")
            print("n_labels: %d, \t n_samples %d, \t n_features %d"
                  % (n_labels, n_samples, n_features))
            print(79 * '_')
            print('% 9s' % 'init'
                  '    time   homo   compl  v-meas     ARI AMI  silhouette')
            Spectral_performance_comparison(data,n_labels)
            print(79 * '_')

            pca = PCA(n_components =10)
            kpca = KernelPCA(n_components =5,kernel ='poly',degree =3)
            fig = k_means_visualise(pca,data,n_labels,y)
            
            plt.title('K-means clustering on the dataset (PCA-reduced data)\n'
                      'Centroids are marked with blue cross')
            
            name = tfilename[0]+'fig' + '_PCA.png'
            fig.savefig(name)   # save the figure to file
            plt.close(fig)
            fig = k_means_visualise(kpca,data,n_labels,y)
            
            plt.title('K-means clustering on the dataset (kPCA-reduced data)\n'
                      'Centroids are marked with blue cross')
            name = tfilename[0]+'fig'+'_kPCA.png'
            fig.savefig(name)   # save the figure to file
            plt.close(fig)

sys.stdout.close()
