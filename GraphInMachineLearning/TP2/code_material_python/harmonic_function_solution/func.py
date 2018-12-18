import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sys
import os

path=os.path.dirname(os.getcwd())
sys.path.append(path)
from helper import *
#from graph_construction.func import *



def build_laplacian_regularized(X, laplacian_regularization ,var=1, eps=0, k=0, laplacian_normalization=""):
    #      a skeleton function to construct a laplacian from data,
    #      needs to be completed
    #
    #  Input
    #  X:
    #      (n x m) matrix of m-dimensional samples
    #  laplacian_regularization:
    #      regularization to add to the laplacian

    # build the similarity graph W
    W = build_similarity_graph(X, var, eps, k)

    #################################################################
    # build the laplacian                                           #
    # L: (n x n) dimensional matrix representing                    #
    #    the Laplacian of the graph                                 #
    # Q: (n x n) dimensional matrix representing                    #
    #    the laplacian with regularization                          #
    #################################################################

    #################################################################
    #################################################################
    return Q











def  mask_labels(Y, l):
    #      a skeleton function to select a subset of labels and mask the rest
    #
    #  Input
    #  Y:
    #      (n x 1) label vector, where entries Y_i take a value in [1..C] (num classes)
    #  l:
    #      number of unmasked (revealed) labels to include in the output
    #
    #  Output
    #  Y_masked:
    #      (n x 1) masked label vector, where entries Y_i take a value in [1..C]
    #           if the node is labeled, or 0 if the node is unlabeled (masked)

    num_samples = np.size(Y,0)

    #################################################################
    # randomly sample l nodes to remain labeled, mask the others    #
    #################################################################


    Y_masked = np.zeros(num_samples)

    i = 0

    while i < l:




    return Y_masked
    #################################################################
    #################################################################
















def hard_hfs(X, Y,laplacian_regularization ,var=1, eps=0, k=0, laplacian_normalization=""):
#  a skeleton function to perform hard (constrained) HFS,
#  needs to be completed
#
#  Input
#  X:
#      (n x m) matrix of m-dimensional samples
#  Y:
#      (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
#
#  Output
#  labels:
#      class assignments for each (n) nodes

    num_samples = np.size(X,0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1
    ####################################################################
    # l_idx = (l x num_classes) vector with indices of labeled nodes   #
    # u_idx = (u x num_classes) vector with indices of unlabeled nodes #
    ####################################################################



    #################################################################
    # compute the hfs solution, remember that you can use           #
    #   build_laplacian_regularized and build_similarity_graph      #
    # f_l = (l x num_classes) hfs solution for labeled.             #
    # it is the one-hot encoding of Y for labeled nodes.            #
    # example:                                                      #
    # if Cl=[0,3,5] and Y=[0,0,0,3,0,0,0,5,5], then f_l is a 3x2    #
    # binary matrix where the first column codes the class '3'      #
    # and the second the class '5'.                                 #
    # In case of 2 classes, you can also use +-1 labels             #
    # f_u = (u x num_classes) hfs solution for unlabeled            #
    #################################################################




    #################################################################
    #################################################################

    #################################################################
    # compute the labels assignment from the hfs solution           #
    # label: (n x 1) class assignments [1,2,...,num_classes]        #
    #################################################################



    #################################################################
    #################################################################
    return labels







def two_moons_hfs():
    # a skeleton function to perform HFS, needs to be completed


    # load the data
    in_data =scipy.io.loadmat(path+'/data/data_2moons_hfs')
    X = in_data['X']
    Y = in_data['Y']

    #################################################################
    # at home, try to use the larger dataset (question 1.2)         #
    #################################################################


    #################################################################
    #################################################################

    # automatically infer number of labels from samples
    num_samples = np.size(X,1)
    num_classes = len(np.unique(Y))

    #################################################################
    # choose the experiment parameter                               #
    #################################################################




    l = 4;# number of labeled (unmasked) nodes provided to the hfs algorithm
    #################################################################
    #################################################################

    # mask labels
    Y_masked =

    #################################################################
    # compute hfs solution using either soft_hfs.m or hard_hfs.m    #
    #################################################################



    #################################################################
    #################################################################

    plot_classification(X, Y,labels,  var=var, eps=0, k=k)
    accuracy = np.mean(labels == np.squeeze(Y))

    return accuracy




















def soft_hfs(X, Y, c_l, c_u, laplacian_regularization ,var=1, eps=0, k=0, laplacian_normalization=""):
    #  a skeleton function to perform soft (unconstrained) HFS,
    #  needs to be completed
    #
    #  Input
    #  X:
    #      (n x m) matrix of m-dimensional samples
    #  Y:
    #      (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
    #  c_l,c_u:
    #      coefficients for C matrix

    #
    #  Output
    #  labels:
    #      class assignments for each (n) nodes

    num_samples = np.size(X,0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    ####################################################################
    # compute the target y for the linear system                       #
    # y = (n x num_classes) target vector                              #
    # l_idx = (l x num_classes) vector with indices of labeled nodes   #
    # u_idx = (u x num_classes) vector with indices of unlabeled nodes #
    ####################################################################

    #################################################################
    #################################################################

    #################################################################
    # compute the hfs solution, remember that you can use           #
    #   build_laplacian_regularized and build_similarity_graph      #
    # f = (n x num_classes) hfs solution                            #
    # C = (n x n) diagonal matrix with c_l for labeled samples      #
    #             and c_u otherwise                                 #
    #################################################################



    #################################################################
    #################################################################

    #################################################################
    # compute the labels assignment from the hfs solution           #
    # label: (n x 1) class assignments [1, ... ,num_classes]        #
    #################################################################

    #################################################################
    #################################################################
    return labels


















def hard_vs_soft_hfs():
    # a skeleton function to confront hard vs soft HFS, needs to be completed

    # load the data
    in_data =scipy.io.loadmat(path+'/data/data_2moons_hfs')
    X = in_data['X']
    Y = in_data['Y']

    # automatically infer number of labels from samples
    num_samples = np.size(X,0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    # randomly sample 20 labels
    l = 20
    # mask labels
    Y_masked =  mask_labels(Y, l)

    Y_masked[Y_masked != 0] = label_noise(Y_masked[Y_masked != 0], 4)

    #################################################################
    # choose the experiment parameter                               #
    #################################################################



    #################################################################
    #################################################################

    #################################################################
    # compute hfs solution using soft_hfs.m and hard_hfs.m          #
    # remember to use Y_masked (the vector with some labels hidden  #
    # as input and not Y (the vector with all labels revealed)      #
    #################################################################



    #################################################################
    #################################################################

    Y_masked[Y_masked == 0] = np.squeeze(Y)[Y_masked == 0]

    plot_classification_comparison(X, Y,hard_labels, soft_labels,var=var, eps=eps, k=k)
    accuracy = [np.mean(hard_labels == np.squeeze(Y)), np.mean(soft_labels == np.squeeze(Y))]
    return accuracy
