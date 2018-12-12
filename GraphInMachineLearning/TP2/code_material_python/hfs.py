import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.io import loadmat
from code_material_python.helper import *

def plot_edges_and_points(X,Y,  W, Y_masked=None, title=''):
    colors=['go-','ro-','co-','ko-','yo-','mo-']
    n=len(X)
    G=nx.from_numpy_matrix(W)
    nx.draw_networkx_edges(G,X)
    for i in range(n):
        plt.plot(X[i,0],X[i,1],colors[int(Y[i])])
    if Y_masked is not None:
        idx = np.arange(len(Y_masked))[(Y_masked > 0).flatten()]
        for i in idx:
            plt.plot(X[i,0],X[i,1],colors[int(Y_masked[i])], markeredgecolor="k")
    plt.title(title)
    plt.axis('equal')

def plot_classification(X, Y,labels, Y_masked=None, var=1, eps=0, k=0):
    plt.figure()

    W = build_similarity_graph(X, var=var, eps=eps, k=k)

    plt.subplot(1, 2, 1)
    plot_edges_and_points(X, Y, W, Y_masked, 'ground truth')
    plt.subplot(1, 2, 2)
    plot_edges_and_points(X, labels, W, Y_masked, 'HFS')

def plot_classification_comparison(X, Y,hard_labels, soft_labels, Y_masked=None, var=1, eps=0, k=0):

    plt.figure()

    W = build_similarity_graph(X, var=var, eps=eps, k=k)

    plt.subplot(1,3,1)
    plot_edges_and_points(X, Y, W, Y_masked=Y_masked, title='ground truth')

    plt.subplot(1,3,2)
    plot_edges_and_points(X, hard_labels, W, Y_masked=Y_masked, title='Hard-HFS')

    plt.subplot(1,3,3)
    plot_edges_and_points(X, soft_labels, W, Y_masked=Y_masked, title='Soft-HFS')

def build_laplacian_regularized(X, laplacian_regularization=0.1,var=1, eps=0, k=0, lap=""):
    W = build_similarity_graph(X, var, eps, k)
    d = np.sum(W, axis=0)
    L = np.diag(d) - W

    if lap == "unn":
        pass
    elif lap == "sym":
        D = np.diag(d ** (-0.5))  # set the diag of D to w
        L = D.dot(L).dot(D)
    elif lap == "rw":
        D = np.diag(d ** (-1))
        L = D.dot(L)
    else:
        raise ArgumentError("The laplacian normalization is unknown.")

    L += laplacian_regularization * np.eye(L.shape[0])
    return L






def mask_labels(y, nb_masked):
    num_samples = y.shape[0]
    y_masked = np.zeros(num_samples)

    while len(np.unique(y))+1 != len(np.unique(y_masked)):
        y_masked = np.zeros(num_samples)
        i = 0
        while i < nb_masked:
            labelled = np.random.randint(1, num_samples, size=nb_masked)
            y_masked[labelled] = y[labelled].flatten()
            i = np.size(np.nonzero(y_masked))
    return y_masked


def hard_hfs(x, y, laplacian_regularization=1.0, var=1.0, eps=0, k=0, lap=""):

    num_samples = x.shape[0]
    cl = np.unique(y)
    l_idx = [idx for idx in range(num_samples) if y[idx]]
    u_idx = [idx for idx in range(num_samples) if not y[idx]]

    f_l = np.array([[y[i] == classe for i in l_idx]
         for classe in cl if classe]).T

    q = build_laplacian_regularized(x, laplacian_regularization, var, eps,
                                    k=k, lap=lap)

    l_uu = q[:, u_idx][u_idx]
    l_ul = q[:, l_idx][u_idx]

    f_u = np.dot(np.linalg.inv(l_uu),
                 -1 * np.dot(l_ul, f_l))

    labels = y.copy()
    labels[u_idx] = [cl[1 + np.argmax(f_u[i])] for i in range(len(u_idx))]
    return labels


# def two_moons_hfs(dataset="./code_material_python/data/data_2moons_hfs",  \
#                   gamma=0.1, var=1, k=5, eps=0, n_l=4, lap="unn"):
#
#     in_data = loadmat(dataset)
#     X = in_data['X']
#     Y = in_data['Y']
#
#     Y_masked = mask_labels(Y, n_l)
#     labels = hard_hfs(X, Y_masked, gamma, var=var, eps=eps, k=k, \
#                       lap=lap)
#
#     W = build_similarity_graph(X, var=var, eps=eps, k=k)
#
#     plot_classification(X, Y,labels, Y_masked,  var=var, eps=eps, k=k)
#
#     return np.mean(labels == np.squeeze(Y))


def label_noise(Y, alpha):
    ind = np.arange(len(Y))
    random.shuffle(ind)
    Y[ind[:alpha]] = 3 - Y[ind[:alpha]]
    return Y

def _compute_binary_matrice(Y):
    num_samples = np.size(Y,0)
    Cl = np.unique(Y).astype(int)
    num_classes = len(Cl)-1
    binary = -np.ones((num_samples, num_classes))
    for i in range(1, len(Cl)):
        subidx = (Y == Cl[i]).flatten()
        binary[subidx, Cl[i]-1] = 1
    return binary

def _compute_C(Y, c_l, c_u):
    C = c_u*np.ones(len(Y))
    C[(Y > 0).flatten()] = c_l
    return np.diag(C)

def _compute_labels(binary, C, K):
    score = inv(inv(C).dot(K) + np.eye(len(C)))
    confidence = score.dot(binary)
    labels = np.argmax(confidence, axis=1) + 1
    return labels

def soft_hfs(X, Y, laplacian_regularization, c_u=1, c_l=1, var=1, eps=0, k=0, lap=""):

    lap = build_laplacian_regularized(X, laplacian_regularization=laplacian_regularization,\
                        var=var, eps=eps, k=k, lap=lap)
    binary = _compute_binary_matrice(Y)
    C = _compute_C(Y, c_l, c_u)
    labels = _compute_labels(binary, C, lap)
    return labels


def two_moons_hfs(dataset="./code_material_python/data/data_2moons_hfs", soft=True,
                   gamma=0.1, var=1, k=5, eps=0, n_l=4, noise=0.):

    in_data = loadmat(dataset)
    X = in_data['X']
    Y = in_data['Y']
    num_samples = np.size(X,1)
    Y_masked = mask_labels(Y, n_l)
    lap = "rw"
    if soft:
        labels = soft_hfs(X, Y_masked, gamma, var=var, eps=eps, k=k, lap=lap)
    else:
        labels = hard_hfs(X, Y_masked, gamma, var=var, eps=eps, k=k, lap=lap)

    plot_classification(X, Y,labels,  var=var, eps=eps, k=k)
    accuracy = np.mean(labels == np.squeeze(Y))

    return accuracy

def hard_vs_soft_hfs(dataset="./code_material_python/data/data_2moons_hfs",
                      gamma=0.1, var=1, k=5, eps=0, n_l=4, c_u=1, c_l=1, noise=False):

    in_data = loadmat(dataset)
    X = in_data['X']
    Y = in_data['Y']
    num_samples = np.size(X,1)
    Y_masked = mask_labels(Y, n_l)
    if noise:
        Y_masked[Y_masked != 0] = label_noise(Y_masked[Y_masked != 0], n_l)
    lap = "rw"

    hard_labels = hard_hfs(X, Y_masked, gamma, var=var, eps=eps, k=k, \
                      lap=lap)
    soft_labels = soft_hfs(X, Y_masked, gamma, var=var, eps=eps, k=k, c_u=c_u,\
                            c_l=c_l, lap=lap)

    plot_classification_comparison(X, Y,hard_labels, soft_labels, Y_masked, var=var, eps=eps, k=k)
    accuracy = [np.mean(hard_labels == np.squeeze(Y)), np.mean(soft_labels == np.squeeze(Y))]
    return accuracy
