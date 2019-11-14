# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:22:38 2019

@author: palan
"""

#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv


# In[70]:


#B = np.random.rand(400,20)
#y = np.random.rand(400,1)


#Calculate Least Square Error
B = np.array([[1,2,3],
            [2,2,4],
            [1,1,1],
            [3,2,2],
            [2,1,2]])

y = np.array([[7],
             [8],
             [3],
             [9],
             [8]])

my_ones = np.ones((len(B),1))
A = np.hstack((my_ones,B))
lss = np.dot(inv(np.dot(A.T,A)),np.dot(A.T,y))
print(lss)


# In[71]:


# Predict Y
x = np.array(lss)
y_predicted = np.dot(A,x)
y_predicted


# In[72]:


#Calculate Mean Square Error
mse = mean_squared_error(y ,y_predicted)
mse


# In[77]:


#Euclidean Distance (K-nearest neighbours) Formula
import math
p1 = [2,0,0]
p2 = [0, 0, 0]
distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)+((p1[2]-p2[2])**2) )

print(distance)


# In[80]:


#dot product of matrix
m1 = np.array([[4.0, 110, 1, 440, 4]])
m2 = np.array([[20],
              [0.07],
              [35],
              [0.01],
              [-10]])

m3 = np.dot(m1,m2)
m3


# In[119]:


# Principal Component Analysis PCA
from numpy import array
from sklearn.decomposition import PCA
# define a matrix
A = array([[1, 2], [0, 1], [-1, -3]])
print("X =",A)
print("\n")
# create the PCA instance
pca = PCA(2)
# fit on data
pca.fit(A)
# access values and vectors
print("U =",pca.components_)
print("\n")
print("Variance")
print(pca.explained_variance_)
print("\n")
# transform data
print("Location of the data point after projection into the loading vector")
B = pca.transform(A)
print(B)
print("original shape:   ", A.shape)
print("transformed shape:", B.shape)


# In[ ]:


#KNN,NB,K-Means


# In[50]:


iter = []
for k in range(0,21):
    iter.append(k)


# In[51]:


from matplotlib import pyplot as plt


# In[52]:


plt.plot(iter,lss)


# In[ ]: