#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:03:09 2019

@author: arjun
"""
import random
import numpy as np
import math
#import time
#import doatools.estimation as estimation 
#import scipy.io as sio
random.seed(10)
f_c = 1e9                           #frequency of operation
w_lambda = 3e8/f_c                  #wavelength
L = 10                               #number of elements
#N= 100                             #number of snapshots
d = 0.5*w_lambda                    #uniform spacing of half wavelength
noise_variance = 0.2             #noise variance 
deviation = 5                       #uncertainty in randomness of nonuniform spacing
array_length = (L-1) * d            #length of the uniform array
m = 1                               #Number of sources
Time = 1
#ts = 1/1e1                         #duration in secs
#t = np.arange(0,Time-ts,ts)        #time snapshoots/music window
N = 1#len(t)                        #number of snapshots received
Iterations = 300000                  #Data size     



X_data = np.zeros(([Iterations,L]),dtype=complex)
Y_data = np.zeros(([Iterations,L]),dtype=complex)
nu_position_data = np.zeros(([Iterations,L]))

X_columns = 2*L
X_D = np.zeros(([Iterations,X_columns]))
Y_D = np.zeros(([Iterations,2*L]))

x_u = np.zeros(([L,N]),dtype=complex) 
x_nu = np.zeros(([L,N]),dtype=complex)

position_x_u = np.linspace(-array_length/2,array_length/2,L)
position_x_u = position_x_u.reshape(L,1) #uniform positions for ULA
position_x_nu = np.linspace(-array_length/2,array_length/2,L)
position_x_nu = position_x_nu.reshape(L,1)
position_x_nu[1:-1] = position_x_nu[1:-1] + (array_length/L) * (np.random.rand(L-2,1) - 0.5)*deviation #nonuniform positions
position_x_nu =  np.sort(position_x_nu,0)

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
    #az_theta_D = np.array([5,30,-25])                          #Deeterministic sequence
    az_theta_D = generate_random_signals(-30,30,m)
    az_theta = az_theta_D*math.pi/180 #DOA to estimate
    phi = 2*math.pi*np.sin(np.tile(az_theta,[L,1]))/w_lambda
    D_u = np.tile(position_x_u,[1,m])
    D_nu = np.tile(position_x_nu,[1,m]) 
    steervec_u = np.exp(1j*phi*D_u)   
    steervec_nu = np.exp(1j*phi*D_nu)                         #uniform steering vectors
    symbols = np.sign(np.random.randn(N,m)) + 1j*np.sign(np.random.randn(N,m)) #QPSK symbols
    nu_position_data[k,:] = position_x_nu.T
    for i in range(N):
        x_u[:,i] = np.sum(np.tile(symbols[i,:],[L,1])*steervec_u,1)  #uniformly sampled data
        x_nu[:,i] = np.sum(np.tile(symbols[i,:],[L,1])*steervec_nu,1)
        noise = noise_variance*np.random.randn(x_nu.shape[0],x_nu.shape[1]) + 1j*noise_variance*np.random.randn(x_nu.shape[0],x_nu.shape[1])      
        #x_u = x_u+noise
        x_nu = x_nu+noise#nonuniformly sampled data
        X_u = x_u.T 
        X_nu = x_nu.T  
        
    X_data[k,:] = X_nu
    Y_data[k,:] = X_u

    
X_data_r = X_data.real
X_data_i = X_data.imag   
 
Y_data_r = Y_data.real
Y_data_i = Y_data.imag   

p=0
for k in range(L):
    X_D[:,p] = X_data_r[:,k]  
    p=p+1
    X_D[:,p] = X_data_i[:,k]
    p=p+1
    #X_D[:,p] = nu_position_data[:,k]
    #p=p+1    

p=0
for k in range(L):
    Y_D[:,p] = Y_data_r[:,k]  
    p=p+1
    Y_D[:,p] = Y_data_i[:,k]
    p=p+1


   

import pickle
#
pickle_out = open("X_data_randsig_aps.pickle","wb")     
pickle.dump(X_D, pickle_out)
pickle_out.close()

pickle_out = open("Y_data_randsig_aps.pickle","wb")     
pickle.dump(Y_D, pickle_out)
pickle_out.close()

pickle_out = open("positions-nu_randsig_aps.pickle","wb")     
pickle.dump(position_x_nu, pickle_out)
pickle_out.close()

#sio.savemat('x_u.mat', {'x_u':x_u})
#sio.savemat('x_nu.mat', {'x_nu':x_nu})

#
#def hermitian(A, **kwargs): ##define hermitian
#    return np.transpose(A,**kwargs).conj()
#H=hermitian
#
#
#r_xx_u = Y_data.T.dot(H(Y_data.T))
#r_xx_nu = X_data.T.dot(H(X_data.T))
#
#estimator = estimation.music.RootMUSIC1D(w_lambda)
#       
## Get the estimates.
#resolved_u, doa_estimate_u = estimator.estimate(r_xx_u, m, unit='deg')
#resolved_nu, doa_estimate_nu = estimator.estimate(r_xx_nu, m, unit='deg')
##estimated DOA in degrees
#est_angles_u = doa_estimate_u.locations
#est_angles_nu = doa_estimate_nu.locations
#print(np.sort(az_theta_D))
#print(est_angles_u)
#print(est_angles_nu)    


