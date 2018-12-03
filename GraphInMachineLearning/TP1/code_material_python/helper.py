# Prim's maximal spanning tree algorithm
# Prim's alg idea:
#  start at any node, find closest neighbor and mark edges
#  for all remaining nodes, find closest to previous cluster, mark edge
#  continue until no nodes remain
#
# INPUTS: graph defined by adjacency matrix, nxn
# OUTPUTS: matrix specifying maximum spanning tree (subgraph), nxn
import matplotlib.pyplot as plt
import scipy
import numpy as np
import networkx as nx







#
# Other routines used: isConnected.m
# GB: Oct 7, 2012


#Copyright (c) 2013, Massachusetts Institute of Technology. All rights
#reserved. Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:

#- Redistributions of source code must retain the above copyright notice, this
#list of conditions and the following disclaimer.
#- Redistributions in binary
#form must reproduce the above copyright notice, this list of conditions and
#the following disclaimer in the documentation and/or other materials provided
#with the distribution.
#- Neither the name of the Massachusetts Institute of
#Technology nor the names of its contributors may be used to endorse or promote
#products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def is_connected(adj,n):
# Uses the fact that multiplying the adj matrix to itself k times give the
# number of ways to get from i to j in k steps. If the end of the
# multiplication in the sum of all matrices there are 0 entries then the
# graph is disconnected. Computationally intensive, but can be sped up by
# the fact that in practice the diameter is very short compared to n, so it
# will terminate in order of log(n)? steps.
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











def plot_edges_and_points(X,Y,W,title=''):
    colors=['go-','ro-','co-','ko-','yo-','mo-']
    n=len(X)
    G=nx.from_numpy_matrix(W)
    nx.draw_networkx_edges(G,X)
    for i in range(n):
        plt.plot(X[i,0],X[i,1],colors[int(Y[i])])
    plt.title(title)
    plt.axis('equal')





def plot_graph_matrix(X,Y,W, title="", log=False):
    plt.figure()
    plt.clf()
    if title != "":
        plt.title(title)
    plt.subplot(1,2,1)
    plot_edges_and_points(X,Y,W)
    plt.subplot(1,2,2)
    if log:
        plt.imshow(np.log(1e-3+W), extent=[0, 1, 0, 1])
    else:
        plt.imshow(W, extent=[0, 1, 0, 1])

    plt.colorbar()
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


def plot_the_bend(X, Y, W, spectral_labels, eigenvalues_sorted, threshold=None):
    plt.figure()
    plt.clf()
    plt.subplot(1,3,1)
    plot_edges_and_points(X,Y,W,'ground truth')

    plt.subplot(1,3,2);
    plot_edges_and_points(X,spectral_labels,W,'spectral clustering')

    plt.subplot(1,3,3);
    if threshold is not None:
        plt.plot([0,len(eigenvalues_sorted)], [threshold, threshold])
    plt.plot(np.arange(0,len(eigenvalues_sorted),1),eigenvalues_sorted,'v:')
    plt.show()
