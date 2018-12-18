import matplotlib.pyplot as plt
import scipy
import numpy as np
import networkx as nx
import random
import cv2 as cv
import scipy.io
import scipy.spatial.distance as sd
import os


def is_connected(adj,n):
    adjn=np.zeros((n,n))
    adji=adj.copy()
    for i in range(n):
        adjn+=adji
        adji=adji.dot(adj)
    return len(np.where(adjn == 0)[0])==0

def max_span_tree(adj):
    n=adj.shape[0]
    if not(is_connected(adj,n)):
        print('This graph is not connected. No spanning tree exists')
    else:
        tr=np.zeros((n,n))
        adj[adj==0]=-np.inf
        conn_nodes = [0]
        rem_nodes = [i+1 for i in range(n-1)]
        while len(rem_nodes)>0:
            L=np.zeros(n)
            L[conn_nodes]=1
            L=L.reshape(n,1)
            C=np.zeros(n)
            C[rem_nodes]=1
            C=C.reshape(1,n)
            B=L.dot(C)
            A=B*adj
            i=np.where(A==np.max(A))[0][0]
            j=np.where(A==np.max(A))[1][0]
            tr[i,j]=1
            tr[j,i]=1
            conn_nodes+=[j]
            rem_nodes.remove(j)
    return tr.astype(int)
            
            
            
def build_similarity_graph(X, var=1, eps=0, k=0):
  assert eps + k != 0, "Choose either epsilon graph or k-nn graph"
  dists = sd.squareform(sd.pdist(X, "sqeuclidean"))
  similarities = np.exp(-dists / var)
  if eps:
    similarities[similarities < eps] = 0
    return similarities
  if k:
    sort = np.argsort(similarities)[:, ::-1]  # descending
    mask = sort[:, k + 1:]  # indices to mask
    for i, row in enumerate(mask): similarities[i, row] = 0
    np.fill_diagonal(similarities, 0)  # remove self similarity
    return (similarities + similarities.T) / 2  # make the graph undirected 
    return similarities

            
     
def build_laplacian(W, laplacian_normalization=""):
  degree = W.sum(1)
  if not laplacian_normalization:
    return np.diag(degree) - W
  elif laplacian_normalization == "sym":
    aux = np.diag(1 / np.sqrt(degree))
    return np.eye(*W.shape) - aux.dot(W.dot(aux))
  elif laplacian_normalization == "rw":
    return np.eye(*W.shape) - np.diag(1 / degree).dot(W)
  else: raise ValueError
  
  
def hardHFS(graph, labels, laplacian):
  classes = np.unique(labels[labels != 0]).reshape((-1, 1))
  f = (labels == classes).astype(np.float).T
  ixmask = np.where(labels == 0)[0]
  ixlabl = np.where(labels != 0)[0]
  luu = laplacian[ixmask][:, ixmask]
  wul = graph[ixmask][:, ixlabl]
  f[ixmask] = np.linalg.pinv(luu).dot(wul.dot(f[ixlabl]))
  return f

              
def plot_edges_and_points(X,Y,W,title=''):
    colors=['go-','ro-','co-','ko-','yo-','mo-']
    n=len(X)
    G=nx.from_numpy_matrix(W)
    nx.draw_networkx_edges(G,X)
    for i in range(n):
        plt.plot(X[i,0],X[i,1],colors[int(Y[i])])
    plt.title(title)
    plt.axis('equal')
                   
def plot_graph_matrix(X,Y,W):
    plt.figure()
    plt.clf()
    plt.subplot(1,2,1)
    plot_edges_and_points(X,Y,W)
    plt.subplot(1,2,2)
    plt.imshow(W, extent=[0, 1, 0, 1])
    plt.show()           
                      
def plot_clustering_result(X,Y,W,spectral_labels,kmeans_labels,normalized_switch=0):
    plt.figure()
    plt.clf()
    plt.subplot(1,3,1)
    plot_edges_and_points(X,Y,W,'ground truth')
    plt.subplot(1,3,2)
    if normalized_switch:
        plot_edges_and_points(X,spectral_labels,W,'unnormalized laplacian')
    else:
        plot_edges_and_points(X,spectral_labels,W,'spectral clustering')
    plt.subplot(1,3,3)
    if normalized_switch:
        plot_edges_and_points(X,kmeans_labels,W,'normalized laplacian')
    else:
        plot_edges_and_points(X,kmeans_labels,W,'k-means')
    plt.show()    

def plot_the_bend(X, Y, W, spectral_labels, eigenvalues_sorted):
    plt.figure()
    plt.clf()
    plt.subplot(1,3,1)
    plot_edges_and_points(X,Y,W,'ground truth')
    plt.subplot(1,3,2)
    plot_edges_and_points(X,spectral_labels,W,'spectral clustering')
    plt.subplot(1,3,3)
    plt.plot(np.arange(0,len(eigenvalues_sorted),1),eigenvalues_sorted,'v:')
    plt.show()         
            
def plot_classification(X, Y,labels,  var=1, eps=0, k=0):
    plt.figure()
    W = build_similarity_graph(X, var=var, eps=eps, k=k)
    plt.subplot(1, 2, 1)
    plot_edges_and_points(X, Y, W, 'ground truth')
    plt.subplot(1, 2, 2)
    plot_edges_and_points(X, labels, W, 'HFS')
    plt.show()         
    
def label_noise(Y, alpha):
    ind=np.arange(len(Y))
    random.shuffle(ind)
    Y[ind[:alpha]] = 3-Y[ind[:alpha]]
    return Y
      
def plot_classification_comparison(X, Y,hard_labels, soft_labels,var=1, eps=0, k=0):
    plt.figure()
    W = build_similarity_graph(X, var=var, eps=eps, k=k)
    plt.subplot(1,3,1)
    plot_edges_and_points(X, Y, W, 'ground truth')
    plt.subplot(1,3,2)
    plot_edges_and_points(X, hard_labels, W, 'Hard-HFS')
    plt.subplot(1,3,3)
    plot_edges_and_points(X, soft_labels, W, 'Soft-HFS')
    plt.show()  


  
    
