import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
import os,sys
import scipy.io as sio
import matplotlib.pyplot as plt

_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64
def hellinger1(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2

def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2

def hellinger3(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

dataset = sio.loadmat('/home/niharika-shimona/Documents/patient_data_red.mat')
X = dataset['patient_data']
[m,n] = X.shape
print X[1,:].shape

hell_dis = np.zeros((m,m))

for i in range(m):
    for j in range(m):
        p =np.histogram(X[i,:],density ='True')
        q =np.histogram(X[j,:],density ='True')
        hell_dis[i,j] = hellinger3(p[0],q[0])

x, y = np.meshgrid(np.linspace(0,1,m), np.linspace(0,1,m))
x, y = x - x.mean(), y - y.mean()
print hell_dis

plt.imshow(hell_dis)
plt.gray()
plt.show()


