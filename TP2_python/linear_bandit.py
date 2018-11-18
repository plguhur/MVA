import numpy as np
from numpy.linalg import inv
from numpy import sqrt
from linearmab_models import ToyLinearModel, ColdStartMovieLensModel
import matplotlib.pyplot as plt
from tqdm import tqdm


### ------------------------------------
### LINEAR BANDIT PROBLEM
### ------------------------------------

def estimate_theta(Z, y, l):
    """ compute theta hat """
    d = Z.shape[1]
    RLS = inv(Z.T.dot(Z) + l * np.eye(d))
    return RLS.dot(Z.T).dot(y)


def lin_ucb(T, model, alpha=2.8, lambda_=1e-3):
    """ Compute a linear UCB algorithm """

    def _find_best_action(features, Z, theta, alpha, l):
        d = features.shape[1]
        bound = np.zeros(n_a)
        for a in range(n_a):
            phi_a = features[a]
            RLS = inv(Z.T.dot(Z) + l * np.eye(d))
            beta = alpha*np.sqrt(phi_a.T.dot(RLS).dot(phi_a))
            bound[a] = phi_a.T.dot(theta) + beta
        return np.argmax(bound)


    n_a = model.n_actions
    d = model.n_features

    theta_hat = np.zeros(d)
    Z = np.zeros((T, d))
    rew = np.zeros(T)
    regret = np.zeros(T)
    norm_dist = np.zeros(T)

    for t in range(T):
        a_t = _find_best_action(model.features, Z[:t], theta_hat, alpha, lambda_)
        rew[t] = model.reward(a_t)
        theta_hat = estimate_theta(Z[:t], rew[:t], lambda_)
        Z[t] = model.features[a_t]
        regret[t] = model.best_arm_reward() - rew[t]
        norm_dist[t] = np.linalg.norm(theta_hat - model.real_theta, 2)

    return norm_dist, regret


def rand_mab(T, model, lambda_=1e-3):
    n_a = model.n_actions
    d = model.n_features
    rew = np.zeros(T)
    regret = np.zeros(T)
    norm_dist = np.zeros(T)
    Z = np.zeros((T, d))

    for t in range(T):
        a_t = np.random.randint(n_a)
        r_t = model.reward(a_t) # get the reward
        rew[t] = r_t
        theta_hat = estimate_theta(Z[:t], rew[:t], lambda_)
        Z[t] = model.features[a_t]
        regret[t] = model.best_arm_reward() - r_t
        norm_dist[t] = np.linalg.norm(theta_hat - model.real_theta, 2)

    return norm_dist, regret


def eps_greedy(T, model, eps=0.1, lambda_=1e-3):
    """ epsilon-greedy algorithm """

    def _find_esp_action(features, theta, eps=0.1):
        return np.random.choice(range(len(features)))  \
            if np.random.random() < eps                \
            else np.argmax(features.dot(theta))

    d = model.n_features
    theta_hat = np.zeros(d)
    Z = np.zeros((T, d))
    regret = np.zeros(T)
    norm_dist = np.zeros(T)
    rew = np.zeros(T)

    for t in range(T):
        a_t = _find_esp_action(model.features, theta_hat, eps=eps)
        rew[t] = model.reward(a_t)
        theta_hat = estimate_theta(Z[:t], rew[:t], lambda_)
        Z[t] = model.features[a_t]
        regret[t] = model.best_arm_reward() - rew[t]
        norm_dist[t] = np.linalg.norm(theta_hat - model.real_theta, 2)

    return norm_dist, regret


def plot_bandit(algorithms, norms, regrets):
    plt.figure(figsize=(20,7))
    plt.subplot(121)
    for i, a in enumerate(algorithms):
        plt.plot(norms[i], label=a['name'])
    plt.ylabel("d(theta, theta_hat)")
    plt.xlabel('Rounds')
    plt.legend()
    plt.grid()


    plt.subplot(122)
    for i, a in enumerate(algorithms):
        plt.plot(regrets[i].cumsum(), label=a['name'])
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Rounds')
    plt.legend()
    plt.grid()
    plt.show()
