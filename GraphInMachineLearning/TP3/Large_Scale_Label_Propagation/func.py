import matplotlib.pyplot as pyplot
import scipy.misc as sm
import numpy as np
import cv2 as cv
import os
import sys
from scipy.spatial import distance
import scipy.io as sio
from tqdm import tqdm, tqdm_notebook

path = os.path.dirname(os.getcwd())
sys.path.append(path)
from helper import *


def hardHFS(graph, labels, laplacian):
  classes = np.unique(labels[labels != 0]).reshape((-1, 1))
  f = (labels == classes).astype(np.float).T
  ixmask = np.where(labels == 0)[0]
  ixlabl = np.where(labels != 0)[0]
  luu = laplacian[ixmask][:, ixmask]
  wul = graph[ixmask][:, ixlabl]
  f[ixmask] = np.linalg.pinv(luu).dot(wul.dot(f[ixlabl]))
  return f




def iterative_hfs(n_iter = 20):

    mat = sio.loadmat("data/data_iterative_hfs_graph.mat")
    W, Y, Y_masked = mat["W"], mat["Y"], mat["Y_masked"]
    classes = np.unique(Y_masked[Y_masked > 0])

    # Compute the initializion vector f
    degrees = np.array(W.sum(0))
    f = (Y_masked == classes).T.astype(np.float)
    n_cols = f.shape[1]
    for _ in tqdm_notebook(range(n_iter)):
        for i in range(n_cols):
            degree = degrees[0][i]
            f[0][i] = np.sum(f[0].dot(W[:,i].toarray()))/degree
            f[1][i] = np.sum(f[1].dot(W[:,i].toarray()))/degree

    labels = np.argmax(f,axis=0)+1
    accuracy = (labels == Y.reshape(-1)).mean()

    return labels, accuracy
