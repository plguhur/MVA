import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
from scipy import stats
from helper import *
from kmeans import *
from scipy.stats import multivariate_normal as mvn


def _em_expectation(X, pi, mu, var):
    N, P = X.shape
    K = len(pi)
    q = np.zeros((K, N))
    for k in range(K):
        for n in range(N):
            gdist = mvn.pdf(X[n], mu[k], var[k])
            q[k, n] = pi[k]*gdist
    acc = np.sum(q, axis=0)
    for k in range(K):    
        q[k, :] /= acc
    return q



def em_fit(X, *params):
    pzx = _em_expectation(X, *params)
    Y = np.argmax(pzx, axis=0)
    return Y


def complete_likelihood(X, pi, mu, var):
    q = _em_expectation(X, pi, mu, var)
    return np.sum(np.log(q))



def _em_maximisation(X, K, q, isotropic=False):
    N, P = X.shape
    mu = np.zeros((K, P))
    var = np.zeros((K, P, P))
    pi = np.zeros(K)
    sum_q = np.sum(q, axis=1)
    
    for k in range(K):
        mu[k] = np.average(X, axis=0, weights=q[k])
        pi[k] = np.mean(q[k])
        if isotropic:
            norm = np.diag((X-mu[k]).dot((X-mu[k]).T))
            v = np.sum(q[k]*norm)/(P*np.sum(q[k])) 
            var[k, ...] = v * np.eye(P)
        else:
            sk = np.zeros((P, P))
            for i in range(N):
                diff = X[i] - mu[k]
                sk += diff.reshape(P,1).dot(diff.reshape(1,P))*q[k,i]
            var[k, ...] = sk/sum_q[k]
    return pi, mu, var
    

def _estimate_params(X, Y, K):
    N, P = X.shape
    mu = np.zeros((K, P))
    Yold = np.zeros(N)
    var = np.zeros((K, P, P))
    pi = np.zeros(K)
    for k in range(K):
        Xk = X[Y == k]
        Nk = len(Xk)
        pi[k] = Nk/N
        mu[k, :] = np.mean(Xk, axis=0)
        var[k, ...] = np.cov(Xk.T)
    return pi, mu, var


def em_train(X, K, vis="all", max_iter=100, isotropic=False):
    N, P = X.shape
    Yold = np.zeros(N)
    colors = np.array([(0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9), (0.5, 0.5, 0.5)], dtype=float)
    llikehood = []
    
    # initialisation
    Y, _ = kmeans(X, K, vis="none", max_iter=100, plusplus=True)
    pi, mu, var = _estimate_params(X, Y, K)
    
    n = 0
    while (Y - Yold).any() != 0 and n < max_iter: 
        n += 1
        Yold = Y.copy()
        
        # expectation
        q = _em_expectation(X, pi, mu, var)
        
        # maximization
        pi, mu, var = _em_maximisation(X, K, q, isotropic=isotropic)
                 
        # clustering
        Y = em_fit(X, pi, mu, var)
        
        # plotting
        if vis == "all":
            plot_em(X, pi, mu, var, colors, n)
        
        # likelihood
        llikehood.append(complete_likelihood(X, pi, mu, var))
        
    return pi, mu, var, llikehood




def plot_em(X, pi, mu, var, colors, n=-1):
    K = len(pi)
    N, P = X.shape
    plt.cla()
    ax = plt.gca()
    plt.axis('equal')
    pzx = _em_expectation(X, pi, mu, var)
    c = pzx.T.dot(colors[:K])
#     for i in range(N):
    plt.scatter(X[:,0], X[:,1], color=c, s=1)
    for k in range(K):
        plt.plot(mu[k,0], mu[k,1], "k*")
        plot_ellipse(mu[k], var[k], ax,alpha=0.3, color=colors[k])
    plt.grid()
    plt.draw()
    if n > 0:
        plt.title(f"EM at step {n}")
    time.sleep(0.1)
    display(plt.gcf(), display_id=True)
    clear_output(wait=True)