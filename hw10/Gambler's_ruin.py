# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 21:00:29 2022

@author: sylee
"""

import numpy as np

p = 18/37
q = 0.55

def phi(X):
    curr = 100
    for i in range(len(X)):
        if X[i]==1:
            curr += 1
        else:
            curr -= 1
        if curr==200:
            return 1
    return 0

def sample(prob, K):
    X=[]
    for i in range(K):
        n = np.random.rand()
        if n<prob:
            X.append(1)
        else:
            X.append(0)
    return X

def f(X, prob):
    answer =1
    for i in range(len(X)):
        if X[i]==1:
            answer *= prob
        else:
            answer *= (1-prob)
    return answer

def estimate(N, K, sampling_prob, real_prob):
    summation = 0
    for _ in range(N):
        X = sample(sampling_prob, K)
        summation += phi(X) * f(X, real_prob)/f(X, sampling_prob)
    return summation/N


print(estimate(3000, 600, q, p))
        


