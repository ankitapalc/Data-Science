# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:42:48 2019

@author: palan
"""

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
#import time

def my_least_squares(B,coeff,n,p):

	A = np.empty((n,p+1))

	A[:,0] = 1
	for i in range(n):
		for j in range(1,p+1):
			A[i,j] = B[i,j-1]

#	ATA = np.dot(np.transpose(A),A)
#	ATAinv = np.linalg.inv(ATA)
#	ATAinvAT = np.dot(ATAinv,np.transpose(A))
#	ATAinvATy = np.dot(ATAinvAT,y)
#	my_solution = np.transpose(ATAinvATy)
	#my_solution = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,y))
	y = np.dot(B,coeff)
	return y

#n= 100
#p = 20
#B=np.random.rand(n,p)
#y=np.random.rand(n,1)

n=5
p=3
B=np.array([[-27,6,3],[10,-3,-5],[-4,3,2]])
coeff = np.array([[6],[-4],[27]])




#start_time = time.clock()
print(my_least_squares(B,coeff,n,p))
#print ("\nExecution time ",time.clock() - start_time, "seconds")




