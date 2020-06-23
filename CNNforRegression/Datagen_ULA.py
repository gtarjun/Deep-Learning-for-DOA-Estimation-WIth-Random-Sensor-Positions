# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:55:41 2019

@author: Arjun
"""

import numpy as np
import math
#import doatools.estimation as estimation 
f_c = 1e9                           #frequency of operation
w_lambda = 3e8/f_c                  #wavelength
L = 8                               #number of elements
#N= 100                              #number of snapshots
d = 0.5*w_lambda                    #uniform spacing of half wavelength
noise_variance = 0.5               #noise variance 
deviation = 4                      #uncertainty in randomness of nonuniform spacing
array_length = (L-1)*d

Time = 1
ts = 1/1e2                        # duration in secs
t = np.arange(0,Time-ts,ts)         #time snapshoots
N = len(t)                          #number of snapshots received
m = 1 #Number of sources
Iterations = 100000

X_data = np.zeros(([Iterations,N,L]),dtype=complex)
Y_data = np.zeros(Iterations)
position_x_u = np.linspace(-array_length/2,array_length/2,L)
position_x_u = position_x_u.reshape(L,1) #uniform positions for ULA

def generate_random_signals(lower_bound, upper_bound, size, scale=None):
   sample_space = (lower_bound + upper_bound)/2
   if scale is None:
      scale = (upper_bound-lower_bound)/2
   results = []
   while len(results) < size:
     samples = np.random.normal(loc=sample_space, scale=scale, size=size-len(results))
     results += [sample for sample in samples if lower_bound <= sample <= upper_bound]
   return np.array(results )   

for k in range(Iterations): 
    #az_theta_D = np.array([5,20,-15]) 
    az_theta_D = generate_random_signals(-60,60,m)
    az_theta = az_theta_D*math.pi/180 #DOA to estimate
    phi = 2*math.pi*np.sin(np.tile(az_theta,[L,1]))/w_lambda
    D_u = np.tile(position_x_u,[1,m])
    steervec_u = np.exp(1j*phi*D_u)                               #uniform steering vectors
    symbols = np.sign(np.random.randn(N,m)) + 1j*np.sign(np.random.randn(N,m)) #QPSK symbols
    x_u = np.zeros(([L,N]),dtype=complex) 
    for i in range(N):
        x_u[:,i] = np.sum(np.tile(symbols[i,:],[L,1])*steervec_u,1)  #uniformly sampled data
        noise = noise_variance*np.random.randn(x_u.shape[0],x_u.shape[1]) + 1j*noise_variance*np.random.randn(x_u.shape[0],x_u.shape[1])  
        x_u = x_u+noise
    X_u = x_u.T
    X_data[k,:,:] = X_u
    Y_data[k] = az_theta 
    
X_data_r = X_data.real
X_data_i = X_data.imag    
    
X_D = np.concatenate((X_data_r,X_data_i))

X_D = X_D.reshape(Iterations,N,L,2)

import pickle

pickle_out = open("X_data_cnn.pickle","wb")     
pickle.dump(X_D, pickle_out)
pickle_out.close()

pickle_out = open("Y_data_cnn.pickle","wb")     
pickle.dump(Y_data, pickle_out)
pickle_out.close()    
            

#def hermitian(A, **kwargs): ##define hermitian
#    return np.transpose(A,**kwargs).conj()
#H=hermitian
#
#
#r_xx_u = x_u.dot(H(x_u))
##r_xx_nu = X_data.T.dot(H(X_data.T))
#
#estimator = estimation.music.RootMUSIC1D(w_lambda)
#       
## Get the estimates.
#resolved_u, doa_estimate_u = estimator.estimate(r_xx_u, m, unit='deg')
##resolved_nu, doa_estimate_nu = estimator.estimate(r_xx_nu, m, unit='deg')
##estimated DOA in degrees
#est_angles_u = doa_estimate_u.locations
##est_angles_nu = doa_estimate_nu.locations
#print(np.sort(az_theta_D))
#print(est_angles_u)
##print(est_angles_nu)    








