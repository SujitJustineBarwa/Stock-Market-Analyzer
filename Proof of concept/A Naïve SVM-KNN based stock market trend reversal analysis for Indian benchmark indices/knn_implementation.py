#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:13:32 2024

@author: justine
"""
import numpy as np

X = np.array([[1,2,3,4,5,6],[1,2,3,4,5,6]]).T

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def knn_search(X_train, X_vec, k):
    predictions = []
    
    # Calculate distances from test point to all training points
    distances = [euclidean_distance(X_vec,train_point) for train_point in X_train]
    
    # Get the indices of the k nearest neighbors
    if k < len(distances):
        k_indices = np.argsort(distances)[:k]
    else:
        k_indices = np.argsort(distances)[:len(distances)]
        
    # Get the labels of the k nearest neighbors
    k_nearest_points = [X_train[i] for i in k_indices]

    return k_nearest_points 

vec = np.array([3.2,3.2]).reshape(1,-1)
a = knn_search(X,vec,10)