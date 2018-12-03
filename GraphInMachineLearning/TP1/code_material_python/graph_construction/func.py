import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))
from helper import *
from graph_construction.generate_data import *


def build_similarity_graph(X, var=1.0, eps=0, k=0):
#      Computes the similarity matrix for a given dataset of samples.
#
#  Input
#  X:
#      (n x m) matrix of m-dimensional samples
#  k and eps:
#      controls the main parameter of the graph, the number
#      of neighbours k for k-nn, and the threshold eps for epsilon graphs
#  var:
#      the sigma value for the exponential function, already squared
#
#
#  Output
#  W:
#      (n x n) dimensional matrix representing the adjacency matrix of the graph
#  similarities:
#      (n x n) dimensional matrix containing
#      all the similarities between all points (optional output)

  assert eps + k != 0, "Choose either epsilon graph or k-nn graph"

  dists = sd.cdist(X, X, 'euclidean')**2
  similarities = np.exp(-dists/var/2) - np.eye(len(X))

  if eps:
    similarities[similarities < eps-1e-6] = 0
    return similarities

  if k:
    i = np.arange(len(similarities))[:, np.newaxis]
    idx = np.argsort(similarities, axis=1)
    similarities = similarities[i, idx]
    similarities[:,:-k] = 0
    undo_idx = np.argsort(idx)
    similarities = similarities[i, undo_idx]
    similarities = np.triu(similarities)+np.triu(similarities).T
    return similarities




def plot_similarity_graph(X, Y, eps=0, var=1, k=0, title="", log=False):
    W = build_similarity_graph(X, var=var, eps=eps, k=k)
    plot_graph_matrix(X,Y,W, title, log=log)



def how_to_choose_epsilon(X, var = 1.0):

    dists = sd.cdist(X, X, 'euclidean')
    similarities = np.exp(-dists/var/2)
    max_tree = max_span_tree(similarities)
    eps = np.min(similarities[max_tree > 0])
    return eps

    # print(f"\epsilon is {eps:.2f}")
