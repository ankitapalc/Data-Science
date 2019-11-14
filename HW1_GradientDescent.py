# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:20:46 2019

@author: palan
"""
import numpy as np
import time

def hypothesis(B,y,n,p,coeff):
	beta1,A = my_least_squares(B,y,n,p)
	beta = coeff
	#print('coeff',beta)
	hx = np.dot(np.dot(np.transpose(A),A),beta)
	return hx

def my_least_squares(B,y,n,p):
	A = np.empty((n,p+1))
	A[:,0] = 1
	for i in range(n):
		for j in range(1,p+1):
			A[i,j] = B[i,j-1]

	ATA = np.dot(np.transpose(A),A)
	ATAinv = np.linalg.inv(ATA)
	ATAinvAT = np.dot(ATAinv,np.transpose(A))
	ATAinvATy = np.dot(ATAinvAT,y)
	beta = np.transpose(ATAinvATy)
	return beta,A

#def cost(B,y,n,p,coeff):
#	total_J = 0.0
##	coeff = np.transpose(coeff)
#	hx = hypothesis(B,y,n,p,coeff)
##	print('hx inside cost',hx)
#	for i in range(len(hx)):
#		total_J = total_J+ (hx[i]-y[i])**2
#	Jx = (1/(2*n))*total_J
##	print('Cost inside cost f',Jx)
#	return Jx


iterations = 1000
alpha = 0.001
def my_gradient_descent(B,y,n,p):
	J_history = list()
	iterarr = list()
	Jx=list()
#	temp = 0
	J1 = 0
	J2 = 0
	total_J = 0.0
	#temp = np.empty((p+1,1))
	beta,A = my_least_squares(B,y,n,p)
	beta = np.transpose(beta)
	hx = np.dot(np.dot(np.transpose(A),A),beta)
#	print('hx',hx)
	cost_term = ((np.dot(np.dot(np.transpose(A),A),beta))-(np.dot(np.transpose(A),y)))**2
	for l in range(len(hx)):
		total_J = total_J+ cost_term[l]
	Jx = (1/(2*n))*total_J
	#print('Initial cost gradient',Jx)
	J_history.append(Jx)
	iterarr.append(0)

	#print('initial co-efficients',beta)
	for j in range(1,iterations):
		J1 = Jx
		firstterm = np.dot(np.dot(np.transpose(A),A),beta)
		secondterm = np.dot(np.transpose(A),y)
#		print('firstterm',firstterm, (firstterm - secondterm),alpha * (firstterm - secondterm),beta)
		reduction_term = (alpha)*(firstterm-secondterm)
		#print('reduction_term',reduction_term)
		for k in range(p+1):
			beta[k] = beta[k] - reduction_term[k]
		#print('gradient beta',beta)
		hx = np.dot(np.dot(np.transpose(A),A),beta)
#		print('after calc hx',hx)
		total_J = 0.0
		cost_term = ((np.dot(np.dot(np.transpose(A),A),beta))-(np.dot(np.transpose(A),y)))**2
		for l in range(len(hx)):
			total_J = total_J+ cost_term[l]
		Jx = (1/(2*n))*total_J
#		Jx = cost(B,y,n,p,beta)
		J2 = Jx
#		print('J1',J1,'J2',J2)
#		print('iterarr',iterarr)
		if (J1<J2):
			break
#		J2 = Jx
		J_history.append(Jx)
		iterarr.append(j)
		print('\niterarr',iterarr)
	#print('Decresing costs',J_history)
	return J1,beta
#	return my_solution

n = 400
p = 100

B=np.random.rand(n,p)
y=np.random.rand(n,1)

start_time_i = time.clock()
my_least_squares(B,y,n,p)
lss_execution = time.clock() - start_time_i
print ("\nLss Execution time ",lss_execution, "seconds")


start_time = time.clock()
my_gradient_descent(B,y,n,p)
gradient_execution = time.clock() - start_time
print ("\nGradient Descent Execution time ",lss_execution+gradient_execution, "seconds")

print('Time difference between both the methods is :',gradient_execution,'seconds\n\n' )


