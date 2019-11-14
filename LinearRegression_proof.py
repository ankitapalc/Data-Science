# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:40:47 2019

@author: palan
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:49:25 2019
@author: palan
"""
# Importing the libraries
import numpy as np

def my_least_squares(B,y):

	A = np.empty((n,p+1))

	A[:,0] = 1
	for i in range(n):
		for j in range(1,p+1):
			A[i,j] = B[i,j-1]

	ATA = np.dot(np.transpose(A),A)
	ATAinv = np.linalg.inv(ATA)
	ATAinvAT = np.dot(ATAinv,np.transpose(A))
	ATAinvATy = np.dot(ATAinvAT,y)
	my_solution = np.transpose(ATAinvATy)
	return my_solution

n= 100
p = 20
B=np.random.rand(n,p);
y=np.random.rand(n,1);

#print(B,y)

print(my_least_squares(B,y));




