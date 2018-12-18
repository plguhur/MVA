import matplotlib.pyplot as pyplot
import scipy.misc as sm
import numpy as np
import cv2 as cv
import os
import sys
from scipy.spatial import distance
import scipy.io as sio
from tqdm import tqdm

path=os.path.dirname(os.getcwd())
sys.path.append(path)
from helper import *






def iterative_hfs(niter = 20):
    # load the data
    # a skeleton function to perform HFS, needs to be completed
    #  Input
    #  niter:
    #      number of iterations to use for the iterative propagation

    #  Output
    #  labels:
    #      class assignments for each (n) nodes
    #  accuracy

    mat = sio.loadmat("../data/data_iterative_hfs_graph.mat")
    W, Y, Y_masked = mat["W"], mat["Y"], mat["Y_masked"]


    classes = np.unique(Y_masked[Y_masked > 0])


    #####################################
    # Compute the initialization vector f #
    #####################################
    

    #####################################
    #####################################

    #################################################################
    # compute the hfs solution, using iterated averaging            #
    # remember that column-wise slicing is cheap, row-wise          #
    # expensive and that W is already undirected                    #
    #################################################################




    ################################################
    # Assign the label in {1,...,c}                #
    ################################################
    labels =

    ################################################
    ################################################
    accuracy = (labels == Y.reshape(-1)).mean()
    return labels, accuracy
