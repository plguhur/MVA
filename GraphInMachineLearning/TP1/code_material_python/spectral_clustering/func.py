import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sklearn.cluster as skc
import sklearn.metrics as skm
from sklearn.cluster import KMeans
import scipy
import sys
import os

path=os.path.dirname(os.getcwd())
sys.path.append(path)
from helper import *
from graph_construction.func import *



def build_laplacian(W, laplacian_normalization=""):
#  laplacian_normalization:
#      string selecting which version of the laplacian matrix to construct
#      either 'unn'normalized, 'sym'metric normalization
#      or 'rw' random-walk normalization

    d = np.sum(W, axis=0)
    L = np.diag(d) - W
    if laplacian_normalization == "unn":
        return L
    elif laplacian_normalization == "sym":
        D = np.diag(d ** (-0.5))  # set the diag of D to w
        return D.dot(L).dot(D)
    elif laplacian_normalization == "rw":
        D = np.diag(d ** (-1))
        return D.dot(L)
    else:
        raise ArgumentError("The laplacian normalization is unknown.")





def spectral_clustering(L, chosen_eig_indices, num_classes=2, debug=False):
#  Input
#  L:
#      Graph Laplacian (standard or normalized)
#  chosen_eig_indices:
#      indices of eigenvectors to use for clustering
#  num_classes:
#      number of clusters to compute (defaults to 2)
#
#  Output
#  Y:
#      Cluster assignments

    E, U = scipy.linalg.eig(L)
    U = U.real
    E = E.real
    idx = E.argsort()
    E =E[idx]
    U = U[:,idx]
    U = U[:, chosen_eig_indices]

    # for debug purpose
    if debug:
        plt.subplot(121)
        plt.plot(U[:,0])
        plt.title("1st selected vector")
        plt.subplot(122)
        plt.plot(np.sort(E), '+')
        plt.title("Sorted eigenvalues")
        plt.show()

    kmeans = KMeans(n_clusters=num_classes, random_state=1231)
    Y = kmeans.fit(U).labels_
    return Y, E




def spectral_clustering_adaptive(L, num_classes=2, debug=False):
    #      a skeleton function to perform spectral clustering, needs to be completed
    #
    #  Input
    #  L:
    #      Graph Laplacian (standard or normalized)
    #  num_classes:
    #      number of clusters to compute (defaults to 2)
    #
    #  Output
    #  Y:
    #      Cluster assignments


    [E,U] = np.linalg.eig(L)
    idx = E.argsort()
    E = E[idx].real
    U = U[:,idx].real
    V = U[:, choose_eig_function(E)]

    # for debug purpose
    if debug:
        plt.subplot(121)
        plt.plot(U[:,0])
        plt.title("1st selected vector")
        plt.subplot(122)
        plt.plot(np.sort(E), '+')
        plt.title("Sorted eigenvalues")
        plt.show()

    kmeans = KMeans(n_clusters=num_classes, random_state=1231)
    Y = kmeans.fit(V).labels_
    return Y, E


def choose_eig_function(eigenvalues):
    #  [eig_ind] = choose_eig_function(eigenvalues)
    #     chooses indices of eigenvalues to use in clustering
    #
    # Input
    # eigenvalues:
    #     eigenvalues sorted in ascending order
    #
    # Output
    # eig_ind:
    #     the indices of the eigenvectors chosen for the clustering
    #     e.g. [1,2,3,5] selects 1st, 2nd, 3rd, and 5th smallest eigenvalues

    eigengaps = np.diff(eigenvalues)
    threshold = abs(eigenvalues[-1])/20
    idx = np.where(eigengaps > threshold)[0]
    if not len(idx):
        return [1]
    return range(1,idx[0]+1)
