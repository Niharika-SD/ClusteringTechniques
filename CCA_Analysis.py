import numpy as np
from time import time
import matplotlib
from SparsePCA_Pipeline import Split_class,evaluate_results
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
import sys,glob,os
from pcp_outliers import pcp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn import metrics
from sklearn.cross_decomposition import CCA
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from pylab import * 
import scipy.io as sio
import pandas as pd

def extract_CCA_dataset(df_aut,df_cont,tasks_list,folder):


	df_aut = df_aut[df_aut[tasks_list[0]]< 7000]
	df_cont = df_cont[df_cont[tasks_list[0]]< 7000]


	y_aut = np.zeros((df_aut.shape[0],len(tasks_list)))
	y_cont = np.zeros((df_aut.shape[0],len(tasks_list)))
	x_cont = np.zeros((1,6670))
	x_aut = np.zeros((1,6670))
	i = 0
	for ID_NO in df_aut['ID']:

				filename = folder + '/Corr_' + `ID_NO` + '.mat'
				data = sio.loadmat(filename) 
				x_aut = np.concatenate((x_aut,data['corr']),axis =0)
				task_order = df_aut[df_aut['ID']==ID_NO].index.tolist()[0]
				for j in range(len(tasks_list)):
					y_aut[i][j] = df_aut[tasks_list[j]][task_order]
				i =i+1

	i=0
	for ID_NO in df_cont['ID']:

				filename = folder + '/Corr_' + `ID_NO` + '.mat'
				data = sio.loadmat(filename) 
				x_cont = np.concatenate((x_cont,data['corr']),axis =0)
				task_order = df_cont[df_cont['ID']==ID_NO].index.tolist()[0]
				for j in range(len(tasks_list)):
					y_cont[i][j] = df_cont[tasks_list[j]][task_order]
				i =i+1

	return x_aut[1:,:],y_aut[:,:],x_cont[1:,:],y_cont[:,:]


if __name__ == '__main__':

	df_aut,df_cont = Split_class()
	tasks_list_ADOS = ['ADOS.RBB','ADOS.SITotal','ADOS.CTotal','ADOS.SCTotal']
	tasks_list_SRS = ["SRS.TotalRaw.Score","SRS.SocAwarRaw.Score","SRS.SocCogRaw.Score","SRS.SocCommRaw.Score","SRS.SocMotRaw.Score","SRS.SocAutManRaw.Score","SRS.SocSRBI.Score"]
	
	tasks_list = tasks_list_SRS
	x_aut,y_aut,x_cont,y_cont = extract_CCA_dataset(df_aut,df_cont,tasks_list,'/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data')

	x = x_aut
	y =y_aut

	L,E,(u,s,v) = pcp(x,'gross_errors', maxiter=1000, verbose=True, svd_method="exact")
	E = E
	L = L
	cca = CCA(scale =False)
	pipeline = Pipeline([('CCA', cca)])

	n_comp = np.asarray(np.linspace(20,40,5),dtype = 'int8')
	p_grid = dict(CCA__n_components = n_comp)

	model =[]
	nested_scores =[] 

	for i in range(30):

		inner_cv = KFold(n_splits=10, shuffle=True, random_state=i)
		outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)

		clf = GridSearchCV(estimator=pipeline, param_grid=p_grid,  cv=inner_cv)
		clf.fit(E,y)

		print clf.best_score_
		print clf.best_estimator_
		print '\n'
		model.append(clf.best_estimator_)

		nested_score = cross_val_score(clf, X=E, y=y, cv=outer_cv)
		print 'mean of nested scores: ', nested_score.mean()
		nested_scores.append(nested_score.mean())
		

	m = nested_scores.index(max(nested_scores))
	final_model = model[m]
 	
 	for i in range(len(tasks_list)):
		
		newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/CCA/'+tasks_list[i]
 		if not os.path.exists(newpath):
				os.makedirs(newpath)
		os.chdir(newpath)
		
		sys.stdout=open('results_'+ tasks_list[i]+'.txt',"w")
		evaluate_results(inner_cv,E,y[:,i],final_model,i)
		sys.stdout.close()

	sio.savemat('lowrank.mat',{'L': L})
	sio.savemat('outliers.mat',{'E': E})