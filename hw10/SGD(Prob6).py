# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 00:41:40 2022

@author: sylee
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

lr = 1e-2
B = 300
iterations = 10000

#log derivative
theta = torch.tensor([0., 0.])
history1 = torch.zeros((iterations+1, 2))
history1[0][1] = 1

for itr in range(iterations):
    mu,tau  = theta[0], theta[1]
    sigma = tau.exp()
    
    X = torch.normal(mu,sigma,size=(B,1))
    g = torch.tensor([mu-1, sigma -1])
    g1 = torch.mean((X * X.sin()) * (X-mu)/sigma**2)
    g2 = torch.mean((X * X.sin()) * (-1+(X-mu)**2/sigma**2))
    g = g+torch.tensor([g1,g2])
    theta -= lr*g
    
    history1[itr+1] = theta
    history1[itr+1][1] = history1[itr+1][1].exp()
print(mu,sigma)

#reparametrization trick
theta = torch.tensor([0., 0.])
history2 = torch.zeros((iterations+1, 2))
history2[0][1] = 1

for itr in range(iterations):
    mu,tau = theta[0], theta[1]
    sigma = tau.exp()
    
    Y = torch.normal(0,1,size=(B,1))
    X = sigma * Y + mu

    g = torch.tensor([mu-1, sigma -1])
    g1 = torch.mean(X.sin() + X * X.cos())
    g2 = torch.mean( (X.sin() + X * X.cos())*Y*sigma )
    g = g+torch.tensor([g1,g2])
    
    theta -= lr*g
    
    history2[itr+1] = theta
    history2[itr+1][1] = history2[itr+1][1].exp()
    
print(mu, sigma)


x1 = np.array(history1[:, 0])
y1 = np.array(history1[:, 1])
x2 = np.array(history2[:, 0])
y2 = np.array(history2[:, 1])
plt.scatter(0,1, s=100, c='green')
plt.plot(x1, y1, linestyle='solid',color='red', label = 'log-derivative')
plt.plot(x2, y2, linestyle='solid',color='blue', label = 'reparametrization')
plt.title('Prob6')
plt.legend()
plt.show()
