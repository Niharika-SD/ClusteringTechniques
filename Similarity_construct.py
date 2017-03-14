import numpy as np
import scipy.io as sio
import os,sys,glob

def E_frob(A,B,n):

	""" Computes the Eros distance between two multi-dimensional timeseries """
	U_A,S_A,V_A = np.linalg.svd(np.dot(A.T,A), full_matrices=True)
	U_B,S_B,V_B = np.linalg.svd(np.dot(B.T,B), full_matrices=True)

	S_1  = (np.linalg.eigvals(np.diag(S_A))).T
	S_2  = (np.linalg.eigvals(np.diag(S_B))).T

	S_1 = S_1.reshape((S_1.shape[0],1))
	S_2 = S_2.reshape((S_2.shape[0],1))

	S = np.concatenate((S_1[0:n],S_2[0:n]),axis =1)
	W = np.mean(S, axis =1)/np.sum(np.mean(S, axis =1))
	
	eros =0 

	for i in range(n):
		eros = eros + W[i]*abs(np.dot(V_A.T[:,i],V_B.T[:,i]))

	dis = np.sqrt(2-2*eros)
	return dis

def main():

	os.chdir('/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data_time_course/')
	
	n_patients = 279
	dis_affinity = np.zeros((n_patients,n_patients))
	for i in range(n_patients):
		for j in range(n_patients):
			print i,j
			data_A = sio.loadmat('patient'+ `i+1` + 'TC.mat')
			data_B = sio.loadmat('patient'+ `j+1` + 'TC.mat')
			A = data_A['patient']['TC_avg'][0,0]
			B = data_B['patient']['TC_avg'][0,0]
			dis_affinity[i][j] = E_frob(A.T,B.T,5)

	for i in range(n_patients):
		dis_affinity[i][i] = 0

	sio.savemat('dis_affinity.mat', {'dis_affinity': dis_affinity})
	


if __name__ == '__main__':
	main()