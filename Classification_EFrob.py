from Similarity_construct import E_frob
import numpy as np 
import scipy.io as sio 
import glob,sys,os

def E_frob2(V_A,V_B,S_1,S_2,n):

	""" Computes the Eros distance between two multi-dimensional timeseries given the right eigenvectors. Modified from Efrob definition """
	
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
	os.chdir('/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data_time_course_com/')
	n_patients = 280

	ID = sio.loadmat('id_corr.mat')
	dis_affinity_IP = np.zeros((n_patients,6670))
	
	for i in range(n_patients):

		aff = []
		data = sio.loadmat('patient'+ `i+1` + '_TC_all.mat')
		for j in range(116):
			A = data['patient']['timeseries'][0,0]['data'][0,j]
			U_A,S_A,V_A = np.linalg.svd(np.dot(A.T,A), full_matrices=True)
			S_1  = (np.linalg.eigvals(np.diag(S_A))).T
			for k in range(j+1,116):	
				
				B = data['patient']['timeseries'][0,0]['data'][0,k]
				U_B,S_B,V_B = np.linalg.svd(np.dot(B.T,B), full_matrices=True)	
				S_2  = (np.linalg.eigvals(np.diag(S_B))).T		
				aff.append(E_frob2(V_A,V_B,S_1,S_2,n=B.shape[1]))

		aff = np.asarray(aff,dtype = np.float32)
		dis_affinity_IP[i,:] =aff.reshape(1,6670)
		sio.savemat('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Affinity/dis_affinity_IP'+`i`+'.mat', {'aff': aff.reshape(1,6670)})
		print('Distances within patient'+`i`+'computed')

	print dis_affinity_IP.shape
	sio.savemat('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Affinity/dis_affinity_IP.mat', {'dis_affinity_IP': dis_affinity_IP})
	


if __name__ == '__main__':
	main()