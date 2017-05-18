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
from pylab import *
from SparsePCA_Pipeline import Split_class,evaluate_results
from sklearn.decomposition import PCA
import scipy.io as sio
import os

def dataset_classification(df_aut,df_cont,folder):

  "Creates a dataset for the classification task"

  x_cont = np.zeros((1,6670))
  x_aut = np.zeros((1,6670))
  
  for ID_NO in df_cont['ID']:

      filename = folder + '/Corr_' + `ID_NO` + '.mat'
      data = sio.loadmat(filename) 
      x_cont = np.concatenate((x_cont,data['corr']),axis =0)

  y_cont = np.zeros((x_cont.shape[0],1))

  for ID_NO in df_aut['ID']:

     filename = folder + '/Corr_' + `ID_NO` + '.mat'
     data = sio.loadmat(filename) 
     x_aut = np.concatenate((x_aut,data['corr']),axis =0)

  y_aut = np.ones((x_aut.shape[0],1))
  
  return x_aut[1:,:],y_aut[1:,:],x_cont[1:,:],y_cont[1:,:]


#module for metric based comparison for kmeans
def bench(estimator, name, data,labels):

  "benchmarker module"
    
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

def bench_spec(estimator, name, data,labels):

  "benchmarker module"
    
  t0 = time()
  estimator.fit(data)
  
  print('% 9s   %.4fs   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

def k_means_performance_comparison(data,n_labels,labels):

  "module for comparing the performance of different clustering runs based on full dataset and the reduced dataset for k means "
  
  bench(KMeans(init='k-means++', n_clusters=n_labels, n_init=10),
	   	name="k-means++", data=data, labels =y)
  bench(KMeans(init='random', n_clusters=n_labels, n_init=10),
		name="random", data=data, labels =y)
	

def spectral_performance_comparison(data,n_labels,labels):

  "module for comparing the performance of different clustering runs based on full dataset and the reduced dataset for spectral clustering"
  print 'NN affinity'

  for i in range (20,40):
    
    print 'no of neighbours = ' + `i`
    bench_spec(SpectralClustering(n_clusters=n_labels,eigen_solver='arpack',affinity="nearest_neighbors",n_neighbors = i),
      name="NN affinity", data=data, labels =y)
    print '\n'

  print 'rbf affinity'
  bench_spec(SpectralClustering(n_clusters=n_labels,eigen_solver='arpack',affinity="rbf"),
    name="rbf", data=data, labels =y)

if __name__ == '__main__':

  df_aut,df_cont = Split_class()
  x_aut,y_aut,x_cont,y_cont = dataset_classification(df_aut,df_cont,'/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data')  
  
  x =np.concatenate((x_cont,x_aut),axis =0)
  y = np.ravel(np.concatenate((y_cont,y_aut),axis =0))

  n_labels = len(np.unique(y))
  newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/Clustering2/'
  if not os.path.exists(newpath):
    os.makedirs(newpath)
  os.chdir(newpath)

  sys.stdout=open('results_spectral'+'.txt',"w")

  print 'raw data'
  spectral_performance_comparison(data = x,n_labels= n_labels, labels=y)

  pca = PCA()

  for i in range(38,130):

      pca1 = PCA(n_components =i)

      data_red = pca.inverse_transform(pca.fit_transform(x)) - pca1.inverse_transform(pca1.fit_transform(x))
      print '\n \n \n '
      print 'rank_solution' + `i`
      spectral_performance_comparison(data_red,n_labels,y)

  sys.stdout.close()



  