# -*- coding: utf-8 -*-
"""
@author: Mhamed JABRI & Oscar CLIVIO
"""

import numpy as np
from Kmeans import KMeans


class EM_GMM():
    

    def __init__(self, k, epsilon=1e-3, init='kmeans', random_seed = 42, nb_iter=20, format_covariance='isotropic'):
        """
        k: the number of clusters
        init : can take one of the two values 'random' or 'kmeans'
        format_covariance: can take one of the two values 'isotropic' or 'general'
        epsilon : Convergence threshold. Iterations when gain on likelihood estimation doesn't exceed epsilon.
        """
        self.k = k
        self.pi = None
        self.mu = None
        self.sigma = None
        self.random_seed = random_seed
        self.epsilon = epsilon
        self.init = init
        self.nb_iter = nb_iter
        self.format_covariance = format_covariance
        self.log_likelihood = []
        self.n_iter = 0
    
    def _init_parameters(self,data) : 
        if self.init == 'random' : 
            self.mu = data[np.random.choice(data.shape[0], self.k, replace=False)]
            self.pi = [1/self.k for j in range(self.k)]
            self.q = 1/self.k * np.ones((data.shape[0],self.k))
        elif self.init == 'kmeans' :
            clf = KMeans(k=self.k, random_seed=self.random_seed, init='kmeans++')
            clf.fit(data)
            self.mu = clf.centers 
            self.pi = [np.sum(clf.labels==j)/data.shape[0] for j in range(self.k)]
            self.q = np.zeros((data.shape[0],self.k))
            for index, label in np.ndenumerate(clf.labels):
                self.q[index,int(label)] = 1
        self.sigma = np.zeros((self.k,data.shape[1],data.shape[1]))
        if self.format_covariance == 'isotropic' :
            for j in range(self.k) : 
                sigma_squared = sum([self.q[i,j]*np.dot(x_i-self.mu[j, :], x_i-self.mu[j, :]) for (i,x_i) in enumerate(data)])/(2*np.sum(self.q[:, j])) 
                self.sigma[j] = sigma_squared * np.identity(data.shape[1])
        elif self.format_covariance == 'general' :
            for j in range(self.k) :
                mu_j = self.mu[j, :].reshape((-1, 1))
                self.sigma[j] = sum([self.q[i,j]*(x_i.reshape((-1,1))-mu_j).dot(x_i.reshape((-1,1)).T-mu_j.T) for (i,x_i) in enumerate(data)])/np.sum(self.q[:, j])                
    
    def multivariate_normal(self,ind_gaussian,data):
        # Computing N(data|self.mu[ind_gaussian,:],self.sigma[ind_gaussian])
        d = data.shape[0]
        denominator = np.sqrt(np.power(2*np.pi,d)*np.linalg.det(self.sigma[ind_gaussian]))
        numerator = np.exp(-0.5*(data-self.mu[ind_gaussian,:].T).T.dot(np.linalg.inv(self.sigma[ind_gaussian])).dot(data-self.mu[ind_gaussian,:].T))
        return (numerator / denominator)

    def _E_step(self, data):
        """
        data: np array of shape (nb_rows, dimension)
        """
        for i in range(data.shape[0]):
            # Computing the common denominator and setting Q_i(z^i =j) to P(z^i=j|x_i)
            denominator = sum([self.pi[j] * self.multivariate_normal(j,data[i,:]) for j in range(self.k)])
            
            for j in range(self.k):
                self.q[i, j] = self.pi[j] * self.multivariate_normal(j,data[i, :]) / denominator

    def _M_step(self, data):
        
        for j in range(self.k):
            #update pi and mu
            self.mu[j, :] = self.q[:, j].T.dot(data) / np.sum(self.q[:, j])
            self.pi[j] = np.mean(self.q[:, j])
            #update sigma
            if self.format_covariance == 'isotropic' : 
                sigma_squared = sum([self.q[i,j]*np.dot(x_i-self.mu[j, :], x_i-self.mu[j, :]) for (i,x_i) in enumerate(data)])/(2*np.sum(self.q[:, j])) 
                self.sigma[j] = sigma_squared * np.identity(data.shape[1])
            elif self.format_covariance == 'general' :
                mu_j = self.mu[j, :].reshape((-1, 1))
                self.sigma[j] = sum([self.q[i,j]*(x_i.reshape((-1,1))-mu_j).dot(x_i.reshape((-1,1)).T-mu_j.T) for (i,x_i) in enumerate(data)])/np.sum(self.q[:, j])


    def _compute_complete_likelihood(self, data):
        sum_i = 0
        for i in range(data.shape[0]):
            sum_j = sum([self.pi[j] * self.multivariate_normal(j,data[i, :]) for j in range(self.k)])
            sum_i += np.log(sum_j)
        return sum_i
    
    
    def fit(self, X):
        self._init_parameters(X)
        log_likelihoods = [self._compute_complete_likelihood(X)]
        gain = np.inf
        while gain > self.epsilon :
            self._E_step(X)
            self._M_step(X)
            log_likelihoods.append(self._compute_complete_likelihood(X))
            gain = np.abs(log_likelihoods[-1] - log_likelihoods[-2])
            self.n_iter += 1
            
            
    def predict_proba(self,data):
        probabilities = np.zeros((data.shape[0], self.k))
        for i in range(data.shape[0]):
            denominator = sum([self.pi[j] * self.multivariate_normal(j,data[i,:]) for j in range(self.k)])
            for j in range(self.k):
                probabilities[i, j] = self.pi[j] * self.multivariate_normal(j,data[i, :]) / denominator
        return probabilities 
    
    
    def predict(self,X) : 
        return np.argmax(self.predict_proba(X), axis=1)
