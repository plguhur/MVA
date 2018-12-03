import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from IPython.display import display, clear_output
import time
from scipy import stats

def generate_blobs(n_samples, n_clusters=2, variance=0, distance=5):
    mu = np.random.rand(n_clusters, 2)*n_clusters*distance
    var = np.eye(2) + np.random.randn(n_clusters, 2, 2)*variance
    x = np.zeros((n_samples, 2))
    y = np.zeros(n_samples)
    bins = n_samples//n_clusters
    for i in range(n_clusters):
        start = i*bins
        end = (i+1)*bins if i + 1 != n_clusters else n_samples
        data = var[i, ...].dot(np.random.randn(2, bins)) + mu[i, :].reshape(2,1)
        x[start:end, :] = data.T
        y[start:end] = i
    return [x, y]


def plot_clusters(X, Y, K):
    for k in range(K):            
        plt.plot(X[Y == k,0], X[Y == k,1], ".")
        
def plot_centroids(centroids):
    for i in range(len(centroids)):
        plt.plot(centroids[i,0], centroids[i,1], "k*")
            
def compute_distortion(X, Y, centroids):
    distortion = 0
    for k in range(len(X)):
        diff = X[k] - centroids[Y[k]]
        distortion += diff.dot(diff)
    return distortion/2
    

def plot_kmeans(X, Y, K, centroids, n=-1):
    plt.axis('equal')
    plot_clusters(X, Y, K)
    plot_centroids(centroids)
    plt.grid()
    plt.draw()
    if n > -1:
        plt.title(f"K-means at step {n}, distortion {compute_distortion(X, Y, centroids):.2f}")
    time.sleep(0.1)
    display(plt.gcf(), display_id=True)
    clear_output(wait=True)


    
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def plot_ellipse(mu, var, ax, alpha=0.3, color=None):
    # https://stackoverflow.com/a/25022642/4986615
    if color is None:
        color = (0.1, 0.2, 0.5)#np.random.random(3)
    vals, vecs = eigsorted(var)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    n_std = 3
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mu, width=width, height=height, angle=theta, color=color)
    ell.set_alpha(alpha=alpha)
    ax.add_artist(ell)


    