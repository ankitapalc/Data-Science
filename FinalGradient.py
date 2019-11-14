#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:30:26 2019

@author: apple
"""
def my_least_squares(B,y):
    ### please put your codes here
    ###
    ###
    one = np.ones((len(B),1))
    A = np.hstack((one , B))
    
    my_solution = np.dot(inv(np.dot(A.T, A)), np.dot(A.T,y))
#    print (my_solution)
    return my_solution

def my_gradient_descent(B,y):
    ### please put your codes here
    ###
    ###
    one = np.ones((len(B),1))
    A = np.hstack((one , B))
    my_solution = np.zeros(len(A[0]))
    init = np.zeros(len(A[0]))
    init = np.array(init)
    for k in range(0,2000):
        d = np.dot(A.T,y)
        s =np.dot(A,init)
        n = np.dot(A.T,s)
        init = init - 0.003*((n- d))
        lss = my_least_squares(B,y)
        if lss.all() < 0.0000000006:
            my_solution = init
            break;      
    return my_solution

import numpy as np
B=np.random.rand(400,5);
y=np.random.rand(400,1);
lss = my_least_squares(B,y)
print (lss)
print("-------------------------------")
print(my_gradient_descent(B,y));