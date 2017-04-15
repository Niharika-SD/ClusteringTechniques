import numpy as np
import scipy.io as sio
import os,sys,glob


def E_frob_eff(V_A,V_B,W):

	""" Computes the Eros distance between two multi-dimensional timeseries given the right eigenvectors and weight matrix """
	
	eros =0 

	for i in range(W.shape[0]):
		eros = eros + W[i]*abs(np.dot(V_A.T[:,i],V_B.T[:,i]))

	dis = np.sqrt(2-2*eros)
	return dis

def calculate_weights(S, method):

	""" Computes the weights from the matrix of eigenvalues using different weighting schemes """

	if method == 'mean':
		 W = np.mean(S, axis =1)/np.sum(np.mean(S, axis =1))

	elif method == 'max':
		 W = np.max(S, axis =1)/np.sum(np.max(S, axis =1))	
		 
	elif method == 'min':
		 W = np.min(S, axis =1)/np.sum(np.min(S, axis =1))

	else :

		W_1 =np.sum(S, axis = 1)
		W_2 = np.dot(W_1.reshape(W_1.shape[0],1) ,np.ones((1,S.shape[1])))
		S_1 = np.divide(S,W_2)

		W = calculate_weights(S_1,'mean')

	return W

def create_affinity_mat(datafolder,n_patients,n_parcellations,n_com):

	""" Computes affinity matrix for each patient """

	os.chdir(datafolder)
	v = np.zeros((1,n_com*n_parcellations))
	s = np.zeros((n_com,1))

	for i in range(n_patients):
			data_A = sio.loadmat('patient'+ `i+1` + 'TC.mat')
			A = data_A['patient']['TC_avg'][0,0]
			[D,V] = np.linalg.eigh(np.dot(A,A.T))
			indx=np.argsort(abs(D))[::-1]
			D=D[indx]
			S = np.sqrt(abs(D))
			V=V[:,indx]
			v = np.concatenate((v,(V[:,0:n_com].T).reshape((1,n_com*n_parcellations))),axis =0)
			s = np.concatenate((s,S[0:n_com].reshape((n_com,1))),axis =1)

	W = calculate_weights(s[:,1:], 'other')
	aff =np.zeros((n_patients,n_patients))

	for i in range(n_patients):
		for j in range(i+1,n_patients):
			V_A = (v[i+1,:].reshape(n_com,n_parcellations)).T
			V_B = (v[j+1,:].reshape(n_com,n_parcellations)).T
			aff[i][j] = E_frob_eff(V_A,V_B,W)
			aff[j][i] = aff[i][j]
			print i, j

	return aff

def create_affinity_mat_IP(datafolder,n_parcellations,patient_ID,n_comp):

	""" Computes affinity matrix for each patient """

	os.chdir(datafolder)	

	for i in range(n_parcellations):

			data_A =sio.loadmat('Ntimeseries_'+ patient_ID + '.mat')
			A = data_A['patient']['timeseries'][0,0]['data'][0,i]

			if i ==0:
				n_com = A.shape[1]
				v = np.zeros((1,n_com*n_com))
				s = np.zeros((n_com,1))
					
			[D,V] = np.linalg.eigh(np.dot(A.T,A))
			indx=np.argsort(abs(D))[::-1]
			D=D[indx]
			S = np.sqrt(abs(D))
			print S
			V=V[:,indx]
			v = np.concatenate((v,(V.T).reshape((1,n_com*A.shape[1]))),axis =0)
			s = np.concatenate((s,S.reshape((S.shape[0],1))),axis =1)

	W = calculate_weights(s[:,1:], 'other')
	print W
	count = 0
	aff = np.zeros((1,((n_parcellations*n_parcellations)-n_parcellations)/2))
	for i in range(n_parcellations):
		for j in range(i+1,n_parcellations):
			V_A = (v[i+1,:].reshape(n_com,A.shape[1])).T
			V_B = (v[j+1,:].reshape(n_com,A.shape[1])).T
			aff[0,count] = E_frob_eff(V_A[:, 1:n_comp],V_B[:,1:n_comp],W[1:n_comp])
			count =count+1
			# print i, j

	return aff

def main():
	iter = sys.argv[1]

	if iter==0:

		datafolder = '/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data_time_course/'
		for i in range(5,117):
			dis_affinity =create_affinity_mat(datafolder,279,116,i)
			sio.savemat('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Comparative_Affinity_n/dis_affinity'+`i`+'.mat', {'dis_affinity': dis_affinity})
	else:

		datafolder = '/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data_timecourse_n_2/'
		dis_affinity_IP = np.zeros((279,6670))
		os.chdir(datafolder)
		i =0
		for file in glob.glob("*.mat"):
			tfilename = file.split('_')[1].split('.')		
			dis_affinity_IP[i] =create_affinity_mat_IP(datafolder,116,tfilename[0],5)
			sio.savemat('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Comparative_Affinity_n/dis_affinity_IP_'+tfilename[0]+'.mat', {'dis_affinity': dis_affinity_IP[i,:]})	
			print 'patient' + tfilename[0]+'processed'
			i =i+1
		sio.savemat('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Comparative_Affinity_n/dis_affinity_IP.mat', {'dis_affinity': dis_affinity_IP[:,:]})

if __name__ == '__main__':
		main()



