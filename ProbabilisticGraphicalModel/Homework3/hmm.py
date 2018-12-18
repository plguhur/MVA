import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, inv, norm
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
        self.precisions = inv(self.covariances) # makes sense for me but does not work
        exponents = np.sum(
            np.einsum('nki,kij->nkj', diff, self.precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(det(self.covariances) * (2 * np.pi) ** self.n_features)
    
    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states):
            #Needs scipy 0.14
            B[s, :] = st.multivariate_normal.pdf(obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)
            #This function can (and will!) return values >> 1
            #See the discussion here for the equivalent matlab function
            #https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/YksWK0T74Ak
            #Key line: "Probabilities have to be less than 1,
            #Densities can be anything, even infinite (at individual points)."
            #This is evaluating the density at individual points...
        return B
    
#     def compute_alpha(self, observations, likelihoods):
#         N = len(observations)
#         alphas = np.zeros((N, self.K))
#         constants = np.zeros(N)
        
#         alpha = self.pi * likelihoods[0]
#         constants[0] = alpha.sum()
#         alphas[0] = alpha / alpha.sum()
#         for i, likelihood in enumerate(likelihoods[1:]):
#             alpha = alphas[i] @ self.transition_matrix * likelihood
#             constants[i+1] = alpha.sum()
#             alphas[i+1] = alpha / alpha.sum()

#         return alphas, constants

    def compute_alpha(self, B):
        log_likelihood = 0.
        T = B.shape[1]
        
        
        alpha = np.zeros(B.shape)
        for t in range(T):
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = B[:, t] * np.dot(self.a.T, alpha[:, t - 1])
         
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, np.ex(alpha
    
    def compute_beta(self, B):
        T = B.shape[1]
        beta = np.zeros(B.shape);
           
        beta[:, -1] = np.ones(B.shape[0])
            
        for t in range(T - 1)[::-1]:
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
        return np.exp(beta)
    
#     def compute_beta(self, observations, likelihoods, constants):
#         N = len(observations)
#         betas = np.zeros((N, self.K))
#         i = N - 1
#         betas[i] = np.ones(self.K)
#         for likelihood, constant in zip(likelihoods[-1:0:-1], constants[-1:0:-1]):
# #             betas[i-1] = self.transition_matrix @ (likelihood * betas[i] / constant) 
#             beta = self.transition_matrix @ (likelihood * betas[i])
#             betas[i-1] = beta / np.sum(beta)
# #          / constant
#             i -= 1

#         return betas

    
    def expectation_step(self, observations):        
#         likelihoods = self.compute_likelihood(observations)
#         alphas, constants = self.compute_alpha(observations, likelihoods)
        s = self.state_likelihood(obs)
        log_likelihood, alpha = self.compute_alpha(s)
        beta = self.compute_beta(s)
        
#         betas = self.compute_beta(observations, likelihoods, constants)
#         posterior = np.divide(alphas * betas, np.sum(alphas*betas, axis=1)[:,None])
        posterior = alphas*betas
        unn_transition = self.transition_matrix * likelihoods[1:, None, :] * betas[1:, None, :] * alphas[:-1, :, None]   
        self.transition_matrix = np.sum(unn_transition, axis=0) / np.sum(unn_transition, axis=(0, 2))
        return posterior
    
    
    # MAXIMIZATION STEP 
    def maximization_step(self, observations, posterior):
        """ Maximization step given the observations, P(q_{t,:}/\theta), P(u_t|u_{t-1})"""
        self.pi = posterior[0] / np.sum(posterior[0])
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
            plt.title(f"Gaussian HMM, step {self.n}")
        self.n += 1
        display(plt.gcf(), display_id=True)
        clear_output(wait=True)
        
    
    
    def fit(self, observations, n_iter=100, plotting="never"):
        """
        perform EM algorithm with Gaussian emission probabilities
        """
        theta_old = norm(self.transition_matrix)
        for i in range(n_iter):
            posterior = self.expectation_step(observations)
            self.maximization_step(observations, posterior)
            
            if plotting == "all" or (plotting == "last" and i == n_iter - 1):
                self.plot(observations, posterior)
                
            theta = norm(self.transition_matrix)
            if np.max(theta - theta_old) < 5e-2:
                break
            theta = theta_old

        if plotting == "last":
            self.plot(observations, posterior)
            
        return posterior

    
    
    
    