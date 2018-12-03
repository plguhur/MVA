import numpy as np
from helper import *

def initialization_plus_plus(X, K):
    N, P = X.shape
    centroids = np.zeros((K, P))
    centroids[0, :] = X[np.random.choice(len(X))]
    for k in range(1, K):
        dist = np.zeros((N, k))
        for kk in range(k):
            diff = X - np.tile(centroids[kk,:], (N, 1))
            L2 = np.sum(diff*diff, axis=1)
            dist[:,kk] = np.sum(diff*diff, axis=1)
        pk = np.min(dist, axis=1)
        pk /= np.sum(pk)
        centroids[k, :] = X[stats.rv_discrete(values=(range(N), pk)).rvs(), :]
    return centroids

def kmeans_fit(X, centroids):
    N, P = X.shape
    K, _ = centroids.shape
    dist = np.zeros((N, K))
    for k in range(K):
        diff = X - np.tile(centroids[k,:], (N, 1))
        dist[:, k] = np.sum(diff*diff, axis=1)
    Ynew = np.argmin(dist, axis=1)
    return Ynew


def kmeans(X, K, vis="all", plusplus=True, max_iter=20):
    N, P = X.shape
    Yold = np.zeros(N)
    Ynew = np.ones(N)
    
    # step 0 : initialisation
    if plusplus:
        centroids = initialization_plus_plus(X, K)
    else:
        Xc = X.copy()
        np.random.shuffle(Xc)
        centroids = Xc[:4, :]
        
    n = 0
    while (Ynew - Yold).any() != 0 and n < max_iter: 
        n += 1
        Yold = Ynew.copy()
        
        # step 1: update distance to centroid
        Ynew = kmeans_fit(X, centroids)

        # step 2: update clusters
        for k in range(K):
            centroids[k, :] = np.mean(X[Ynew == k, :], axis=0)

        # plotting
        if vis == "all":
            plot_kmeans(X, Ynew, K, centroids, n)
            
    return Ynew, centroids