import numpy as np
from numpy.linalg import inv
from numpy import sqrt, log
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from IPython.display import display, clear_output

### ------------------------------------
### MULTI ARM BANDIT
### ------------------------------------
def _update_empirical_mean(mu, value, N):
    ''' Online update of the emperical mean '''
    return N/(N+1) * mu + 1/(N+1) * value


def _update_ucb(B, mu, N, t, rho=sqrt(3)):
    K = len(B)
    for i in range(K):
        B[i] = mu[i] + rho*sqrt(log(t)/N[i])


def plot_UCB1(B, mu, gt=None, sleep=0.1, title="UCB1"):
    K = len(mu)
    plt.cla()
    ax = plt.gca()
    plt.errorbar(x=range(K), y=mu, yerr=(B-mu), fmt='o')
    plt.grid()
    plt.ylim(-5, 5)
    if gt is not None:
        plt.scatter(range(0, K), gt, color="k", label="ground truth")
    plt.draw()
    plt.xlabel("Arm")
    plt.ylabel("Distribution")
    plt.title(title)
#     time.sleep(sleep)
    display(plt.gcf(), display_id=True)
    clear_output(wait=True)


def UCB1(T, MAB, rho=sqrt(3), vis="all"):
    """The UCB1 algorithm starts with an initialization phase that draws each arm once, and for t ≥ K, chooses at time t + 1 arm
    $$A_{t+1} = argmax_{a\in{1,...,K }} = \hat{\mu}_a(t) + \rho_t \sqrt{\frac{\log t}{2N_a(t)}}$$
    """
    K = len(MAB) # number of arms
    mu = np.zeros(K) # empirical mean
    N = np.zeros(K) # number of times an arm was pulled
    B = np.zeros(K) # upper-confidence bound
    gt = [m.mean for m in MAB] # ground-truth
    draws = np.zeros(T+1)
    rew = np.zeros(T+1)

    # initialisation
    for i in range(K):
        s = MAB[i].sample()
        mu[i] = _update_empirical_mean(0, s, 0)
        B[i] = mu[i] + 0
        N[i] += 1
        draws[i] = i
        rew[i] = s

    # steps
    try:
        for t in range(K, T+1):
            _update_ucb(B, mu, N, t, rho=rho)
            a = np.argmax(B)
            s = MAB[a].sample()
            mu[a] = _update_empirical_mean(mu[a], s, N[a])
            N[a] += 1
            draws[t] = a
            rew[t] = s

            if vis == "all":
                plot_UCB1(B, mu, gt=gt, sleep=1e-3)
    except KeyboardInterrupt:
        pass

    return rew, draws, mu

def _update_posterior(p, a, b):
    K = len(p)
    for i in range(K):
        p[i] = np.random.beta(a[i], b[i])

def plot_TS(a, b, gt=None, size=1000, sleep=1e-2):
    K = len(a)
    plt.cla()
    ax = plt.gca()
    all_data = [np.random.beta(a[i], b[i], size=size) for i in range(K)]
    plt.violinplot(all_data, showmeans=True, showmedians=True)
    if gt is not None:
        plt.scatter(range(1, K+1), gt, color="k", label="ground truth")
    plt.grid()
    plt.ylim(-0.5, 1.5)
    plt.xlabel("Arm")
    plt.ylabel("Distribution")
    plt.draw()
    plt.title("Thomson Sampling")
    time.sleep(sleep)
    display(plt.gcf(), display_id=True)
    clear_output(wait=True)


def TS(T, MAB, rho=sqrt(3), vis="all"):
    """Thomson sampling algorithm for Bernoulli distribution
    """
    K = len(MAB) # number of arms
    p = np.random.random(K) # prior distribution
    a = np.ones(K) # beta - a
    b = np.ones(K) # beta - b
    gt = [m.mean for m in MAB] # ground-truth
    draws = np.zeros(T+1)
    rew = np.zeros(T+1)

    # steps
    try:
        for t in range(1, T+1):
            _update_posterior(p, a, b)
            A = np.argmax(p)
            s = MAB[A].sample()
            a[A] += s
            b[A] += 1-s
            draws[t] = A
            rew[t] = s

            if vis == "all":
                plot_TS(a, b, gt=gt, sleep=1e-3)
    except KeyboardInterrupt:
        pass

    return rew, draws, p


def plot_naive(mu, gt=None, sleep=0.1):
    K = len(mu)
    plt.cla()
    ax = plt.gca()
    plt.grid()
    plt.ylim(-5, 5)
    plt.scatter(range(K), mu, label="estimate $\mu$")
    if gt is not None:
        plt.scatter(range(K), gt, color="k", label="ground truth")
    plt.draw()
    plt.xlabel("Arm")
    plt.ylabel("Distribution")
    plt.title("Naive")
    time.sleep(sleep)
    display(plt.gcf(), display_id=True)
    clear_output(wait=True)


def naive(T, MAB, vis="all"):
    """Naive implementation that selects only the best arm
    """
    K = len(MAB) # number of arms
    mu = np.zeros(K) # empirical mean
    N = np.zeros(K) # number of times an arm was pulled
    B = np.zeros(K) # upper-confidence bound
    gt = [m.mean for m in MAB] # ground-truth
    draws = np.zeros(T+1)
    rew = np.zeros(T+1)

    # initialisation
    for i in range(K):
        s = MAB[i].sample()
        mu[i] = _update_empirical_mean(0, s, 0)
        B[i] = mu[i] + 0
        N[i] += 1
        draws[i] = i
        rew[i] = s

    # steps
    try:
        for t in range(K, T+1):
            a = np.argmax(mu)
            s = MAB[a].sample()
            mu[a] = _update_empirical_mean(mu[a], s, N[a])
            N[a] += 1
            draws[t] = a
            rew[t] = s

            if vis == "all":
                plot_naive(mu, gt=gt, sleep=1e-3)
    except KeyboardInterrupt:
        pass

    return rew, draws, mu



def expected_regret(mab, simulator, T=200, n_iter=100):
    cum_rew = np.zeros((n_iter, T+1))
    gt = [m.mean for m in mab]
    opt = np.max(gt)

    for i in range(n_iter):
        rew, draws, p = simulator(T, mab, vis="never")
        cum_rew[i, :] = np.cumsum(rew)

    exp_regrets = np.zeros(T)
    for t in range(T):
        exp_regrets[t] = t * opt - np.mean(cum_rew[:, t])

    return exp_regrets



def Kullback_Leibler_2(x, y):
    return x*log(x/y) + (1-x) * log((1-x)/(1-y))

def oracle(mab):
    gt = [m.mean for m in mab]
    star = np.argmax(gt)
    p_star = gt[star]
    pa = [m for i, m in enumerate(gt) if i != star]
    C = np.sum([(p_star - p)/Kullback_Leibler_2(p, p_star) for p in pa])
    return C


def TS2(T, MAB, rho=sqrt(3), vis="all"):
    """
    Thomson sampling handling non-binary rewards [Agrawal and Goyal, 2012]
    """
    K = len(MAB) # number of arms
    p = np.random.random(K) # prior distribution
    a = np.ones(K) # beta - a
    b = np.ones(K) # beta - b
    gt = [m.mean for m in MAB] # ground-truth
    draws = np.zeros(T+1)
    rew = np.zeros(T+1)

    # steps
    for t in range(1, T+1):
        _update_posterior(p, a, b)
        A = np.argmax(p)
        s = MAB[A].sample()
        s_ber = np.random.random() < s
        a[A] += s_ber
        b[A] += 1-s_ber
        draws[t] = A
        rew[t] = s

        if vis == "all":
            plot_TS(a, b, gt=gt, sleep=1e-3)

    return rew, draws, p
