import numpy as np
import sklearn.datasets as skd

def worst_case_blob(num_samples, gen_pam, blob_var=0.3):

    X, Y = skd.make_blobs(n_samples=num_samples, n_features=2, centers=np.column_stack((0,0)), cluster_std=blob_var)
    X[-1]=[np.max(X)+gen_pam,0]
    return [X,Y]


def blobs(num_samples,nBlobs,blob_var,surplus=0):
#   Creates N gaussian blobs evenly spaced across a circle
#
#  Input
#  num_samples:
#      number of samples to create in the dataset
#
#   how many separate blobs to create,
#   gaussian variance of each blob,
#   surplus of samples added to first blob to create unbalanced classes
#
#  Output
#  X:
#      (n x m) matrix of m-dimensional samples
#  Y:
#      (n x 1) matrix of "true" cluster assignment [1:c]
    x = np.arange(nBlobs) * 2 * np.pi / nBlobs
    xf=x[0]
    x=x[1:]
    centers = np.column_stack((np.cos(x), np.sin(x)))
    centerf= np.column_stack((np.cos(xf), np.sin(xf)))
    Xf, Yf = skd.make_blobs(n_samples=int((num_samples-surplus)/nBlobs+surplus), n_features=2, centers=centerf, cluster_std=blob_var)
    X, Y = skd.make_blobs(n_samples=int((num_samples-surplus)/nBlobs), n_features=2, centers=centers, cluster_std=blob_var)
    X=np.vstack((Xf, X))
    Y=np.array(list(Yf)+list(Y+1))
    return [X,Y]



def two_moons(num_samples,moon_radius,moon_var):
#   Creates two intertwined moons
#
#  Input
#  num_samples:
#      number of samples to create in the dataset
#  radius of the moons,
#  variance of the moons
#
#  Output
#  X:
#      (n x m) matrix of m-dimensional samples
#  Y:
#      (n x 1) matrix of "true" cluster assignment [1:c]
    X=np.zeros((num_samples,2))

    for i in range(int(num_samples/2)):
        r = moon_radius + 4*i/num_samples
        t = i*3/num_samples*np.pi
        X[i, 0] = r*np.cos(t)
        X[i, 1] = r*np.sin(t)
        X[i + int(num_samples/2), 0] = r*np.cos(t + np.pi)
        X[i + int(num_samples/2), 1] = r*np.sin(t + np.pi)


    X= X + np.sqrt(moon_var) * np.random.normal(size=(num_samples, 2))
    Y = np.ones(num_samples)
    Y[:int(num_samples/2)+1]=0
    return [X,Y.astype(int)]
