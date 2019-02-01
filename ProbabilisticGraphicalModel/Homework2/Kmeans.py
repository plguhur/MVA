# -*- coding: utf-8 -*-
"""
@author: Mhamed JABRI & Oscar CLIVIO
"""

import numpy as np
from scipy.spatial.distance import cdist


class KMeans():
    
    
    def __init__(self, k, epsilon=1e-5, random_seed=42, init="generic"):
        self.k = k
        self.centers = None
        self.labels = None
        self.epsilon = epsilon
        self.init = init 
        self.random_seed = random_seed
        self.iterations_before_cv = 0
        
        
    def fit(self, X):
        # reproduce results
        np.random.seed(self.random_seed)
        # initialize the centroids 
        if self.init == "generic":
            self.centers = X[np.random.choice(X.shape[0], self.k, replace=False)]
            
        elif self.init == 'kmeans++':
            centers = np.zeros((self.k, X.shape[1]))
            centers[0] = X[np.random.choice(X.shape[0], 1)]
            for j in range(1,self.k) : 
                distance_matrix = cdist(X,np.array(centers))
                distances_to_nearest_clusters = np.square(np.amin(distance_matrix, axis=1))
                centers[j]=X[np.random.choice(X.shape[0], 1, p=distances_to_nearest_clusters/sum(distances_to_nearest_clusters))]
            self.centers = centers
                
        elif self.init not in ["generic","kmeans++"]:
            return print("The two possible options to initialize the centroids are 'generic' and 'kmeans++'")
            
        #Compute the centers of the clusters and assign the data
        centers = None
        while (centers is None or np.abs(centers - self.centers).max() > self.epsilon):
            centers = self.centers.copy()
            distance_matrix = cdist(X,self.centers)
            #Associate each xi to the nearest center μk.
            assigned_clusters = np.argmin(distance_matrix, axis=1)
            #Compute the new centers.
            X_ = np.concatenate((X, assigned_clusters.reshape(-1,1)), axis=1)
            self.centers = np.array([np.mean(X_[X_[:,-1]==j][:,0:-1], axis=0) for j in range(self.k)])
            #Compute number of iterations before convergence
            self.iterations_before_cv += 1
            
        self.labels = X_[:,-1]
    
    
    def compute_distortion(self,X):
        """
        Compute the following :
        J = \sum_{data point}\sum_{nb clusters} z_{i}^k ||x_i - \mu_{k}||² 
        where z_{i}^k is equal to 1 if the point x_i is assigned to cluster k 
        """
        X_ = np.concatenate((X, self.labels.reshape(-1,1)), axis=1)
        J = sum([np.sum(np.square(np.linalg.norm(X_[X_[:,-1]==j][:,0:-1] - self.centers[j], axis=1))) for j in range(self.k)])
        return J
    
    
    def predict(self,X):
        distance_matrix = cdist(X,self.centers)
        assigned_clusters = np.argmin(distance_matrix, axis=1)
        return assigned_clusters 
   

