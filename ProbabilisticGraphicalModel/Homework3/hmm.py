import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, inv
from matplotlib.collections import EllipseCollection
from IPython.display import clear_output, display


from scipy.misc import logsumexp


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]
    
class GaussianHiddenMarkovModel():

    def __init__(self, pi0, a0, means, covariances):
        self.n_features = means.shape[1]
        self.K = len(pi0)
        self.pi = pi0
        self.transition_matrix = a0
        self.means = means
        self.covariances = covariances
        self.precisions = inv(covariances)
        self.n = 0

    # EXPECTATION STEP
    def compute_likelihood(self, observations):
        diff = observations[:, None, :] - self.means
#         exponents = np.sum(
#             np.einsum('nki,kij->nkj', diff, inv(self.covariances)) * diff, axis=-1)
        exponents = np.sum(
            np.einsum('nki,kij->nkj', diff, self.precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(det(self.covariances) * (2 * np.pi) ** self.n_features)
        
    def compute_alpha_beta(self, observations):
        """
        Implementation of the $\alpha$ and  $\beta$ recusions to estimate posterior distributions of hidden states
        """        
        N = len(observations)
        likelihoods = self.compute_likelihood(observations)
        alphas = np.zeros((N, self.K))
#         betas = np.zeros((N, self.K))
        constants = np.zeros(N)
        
        alpha = self.pi * likelihoods[0]
        constants[0] = alpha.sum()
        alphas[0] = alpha / alpha.sum()
        for i, likelihood in enumerate(likelihoods[1:]):
            alpha = alphas[i] @ self.transition_matrix * likelihood
            constants[i+1] = alpha.sum()
            alphas[i+1] = alpha / alpha.sum()

        beta = np.ones(self.K)
        betas = [beta]
        for likelihood, constant in zip(likelihoods[-1:0:-1], constants[-1:0:-1]):
            beta = self.transition_matrix @ (likelihood * betas[0]) / constant
            betas.insert(0, beta)

        return alphas, np.asarray(betas)

    
    def expectation_step(self, observations):        
        likelihoods = self.compute_likelihood(observations)
        alphas, betas = self.compute_alpha_beta(observations)
        posterior = alphas * betas #/ np.sum(alphas*betas)
        transition = self.transition_matrix * likelihoods[1:, None, :] * betas[1:, None, :] * alphas[:-1, :, None]   
        return posterior, transition
    
    
    # MAXIMIZATION STEP 
    def maximization_step(self, observations, posterior, proba_transition):
        """ Maximization step given the observations, P(q_{t,:}/\theta), P(u_t|u_{t-1})"""
        self.pi = posterior[0] / np.sum(posterior[0])
        self.transition_matrix = np.sum(proba_transition, axis=0) / np.sum(proba_transition, axis=(0, 2))
        Nk = np.sum(posterior, axis=0)
        self.means = (observations.T @ posterior / Nk).T
        diffs = observations[:, None, :] - self.means
        self.covariances = np.einsum('nki,nkj->kij', diffs, diffs * posterior[:, :, None]) / Nk[:, None, None]
        return self.pi, self.transition_matrix, self.means, self.covariances

    # PLOT
    def plot_ellipses(self, ax, alpha=0.3, cmap=None, n_std=3):
        widths, heights, thetas = np.zeros(self.K), np.zeros(self.K), np.zeros(self.K)
        for k in range(self.K):
            vals, vecs = eigsorted(self.covariances[k])
            thetas[k] = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            widths[k], heights[k] = 2 * n_std * np.sqrt(vals)
        ec = EllipseCollection(widths, heights, thetas, units='x', offsets=self.means,
                           transOffset=ax.transData)
        ec.set_array(np.arange(self.K))
        if cmap is not None:
            ec.set_cmap(cmap)
        ec.set_alpha(alpha=alpha)
        ax.add_collection(ec)

    def plot(self, observations, posteriors):
        plt.cla()
        ax = plt.gca()
        plt.axis('equal')
        colors = np.argmax(posteriors, axis=-1)
        p = plt.scatter(observations[:, 0], observations[:, 1], c=colors,  s=1)
        cmap = p.get_cmap()
        plt.plot(self.means[:, 0], self.means[:,1], "r*")
        self.plot_ellipses(ax, alpha=0.3, cmap=cmap)
        plt.xlabel("x1")
        plt.xlabel("x2")
        plt.grid()
        plt.draw()
        if self.n > 0:
#             loss = self.estimate_log_likelihood(observations, posteriors)
#             plt.title(f"Gaussian HMM, loss {loss}")
            plt.title(f"Gaussian HMM, step {self.n}")
        self.n += 1
        display(plt.gcf(), display_id=True)
        clear_output(wait=True)
        
    
    
    def fit(self, observations, n_iter=100, plotting="never"):
        """
        perform EM algorithm with Gaussian emission probabilities
        """
        theta_old = np.hstack((self.pi.ravel(), self.transition_matrix.ravel()))
        for i in range(n_iter):
            posterior, transition = self.expectation_step(observations)
            self.maximization_step(observations, posterior, transition)
            
            if plotting == "all" or (plotting == "last" and i == n_iter - 1):
                self.plot(observations, posterior)
                
            theta = np.hstack((self.pi.ravel(), self.transition_matrix.ravel()))
            if np.allclose(theta, theta_old):
                break
            else:
                theta = theta_old
#             theta = np.hstack((self.pi.ravel(), self.transition_matrix.ravel()))
#             if np.max(theta_old - theta) < 1e-2:
#                 break
#             theta_old = theta.copy()
            
        if plotting == "last":
            self.plot(observations, posterior)
            
        forwards, backwards = self.compute_alpha_beta(observations)
        posteriors = np.asarray(forwards) * np.asarray(backwards)
        return posteriors

    
    
    
    