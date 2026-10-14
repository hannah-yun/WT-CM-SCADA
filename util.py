import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.special import loggamma, digamma
from scipy.sparse.linalg import eigs
from sklearn.metrics import auc

### NOTE: Difference between loggamma() and gammaln() ###
#   loggamma(negative) = Nan
#   gammaln(negative) = inf
#   loggamma(complex) = complex
#   gammaln(complex) = undefined
### Dirichlet.py chooses gammaln(). Need real vectors ###


err = 1e-7


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plinear_approximate_sigmoid(x):
    return (x / 4 + 0.5) * (x > 0) + (0.5 - x / 4) * (x > 1)


def ReLu(x):
    return x * (x > 0)


def pReLu(x):
    return x * (x > 0) + (1 - x) * (x > 1)


def ptanh(x):
    return (np.tanh(x) + 1) / 2


# Stationary vector for Markov Chain
def fast_markov_vector(M):
    vec = np.real(eigs(M.astype(float), 1)[1][:, 0])
    return np.real(vec * np.conj(vec))

def fast_markov_vector_robust(M, max_iter=2000, tol=1e-10, damping=1e-6):
    """
    Robust stationary distribution for a row-stochastic transition matrix M.

    - Uses power iteration: pi = pi M
    - Adds tiny damping/teleport to guarantee ergodicity and avoid convergence issues.
    - Returns a proper probability vector (nonnegative, sums to 1).
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]

    # sanitize
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    # ensure row-stochastic
    row_sum = M.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    M = M / row_sum

    # make ergodic (prevents periodicity / reducibility issues)
    J = np.ones((n, n), dtype=float) / n
    M = (1.0 - damping) * M + damping * J

    # power iteration on left eigenvector
    pi = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        pi_next = pi @ M
        if np.linalg.norm(pi_next - pi, 1) < tol:
            pi = pi_next
            break
        pi = pi_next

    # normalize and clip
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s <= 0:
        pi = np.ones(n, dtype=float) / n
    else:
        pi = pi / s
    return pi
    

def markov_vector(M, non_zero=True):
    values, vectors = np.linalg.eig(M)

    idx = np.argmin(abs(values - 1))
    vec = np.real(vectors[:, idx] * np.conj(vectors[:, idx]))
            
    # Pad 0 with a small number (for MLE)
    if non_zero:
        # min value
        minval = np.min(vec[np.nonzero(vec)])
        minval = min(minval / 10 + (minval == 0), minval, err / 10)
        vec[np.nonzero(vec == 0)] = minval
                
    return vec


def norm_transition(transition):        
    # Remove diagonal elements
    np.fill_diagonal(transition, 0.)

    normalization = transition.sum(axis=0, keepdims=False)
    # Check 0 in case the term stay in diagonal
    for i in np.nonzero(normalization == 0)[0]:
        normalization[i] = 1.
        try:
            transition[:, i] = 1. / (transition.shape[0] - 1)
            transition[i, i] = 0.
        # Divide by 0. 1 point only. Theoretically impossible
        except:
            transition[:, i] = 1.

    return transition / normalization


def inner_transition(emb, active_function=sigmoid):
    transition = active_function(np.dot(emb, emb.T))
    
    return norm_transition(transition)


def kl_dirichlet(alpha, beta):
    alpha_0 = sum(alpha)
    beta_0 = sum(beta)
    diff = alpha - beta
    return loggamma(alpha_0) - loggamma(beta_0) \
          - sum(loggamma(alpha)) + sum(loggamma(beta)) \
          + sum(diff * digamma(alpha)) \
          - digamma(alpha_0) * sum(diff)


def reverse_kl_dirichlet(alpha, beta):
    return kl_dirichlet(beta, alpha)


def jeffreys_dirichlet(alpha, beta):
    alpha_0 = sum(alpha)
    beta_0 = sum(beta)
    diff = alpha - beta
    return sum(diff * (digamma(alpha) - digamma(beta))) \
          - (digamma(alpha_0) - digamma(beta_0)) * sum(diff)


# Normalized term ignored
def nml_code_length(alpha, window_size):
    alpha_0 = sum(alpha)
    return window_size * (sum(loggamma(alpha)) - loggamma(alpha_0) - sum((alpha - 1) * (digamma(alpha) - digamma(alpha_0))))


# Evaluation method. Implemented according to SMDL paper
def auc_benefit_fan(scores, change_points, tolerate=100, threshold_step=0.0001):
    change_points = np.array(change_points)

    threshold = 0.

    benefits = []
    false_alarms = []

    while threshold <= 1.:
        benefit = 0.
        false_alarm = 0.
        for time in range(len(scores)):
            if scores[time] > threshold:
                predict = np.min(np.abs(change_points - time))
                if predict < tolerate:
                    benefit += 1 - predict / tolerate
                else:
                    false_alarm += 1

        benefits.append(benefit)
        false_alarms.append(false_alarm)

        threshold += threshold_step

    if np.max(benefits) != 0:
        benefits /= np.max(benefits)
    if np.max(false_alarms) != 0:
        false_alarms /= np.max(false_alarms)
    
    return auc(false_alarms, benefits)


# Evaluation method. Implemented according to TREE paper (Graph Partitioning)
def auc_benefit_far(scores, change_points, tolerate=3, threshold_step=0.0001):
    change_points = np.array(change_points)

    threshold = 0.

    benefits = []
    false_alarms = []

    while threshold <= 1.:
        benefit = 0.
        false_alarm = 0.
        total_alarm = 0.
        for time in range(len(scores)):
            if scores[time] > threshold:
                total_alarm += 1
                predict = np.min(np.abs(change_points - time))
                if predict < tolerate:
                    benefit += 1 - predict / tolerate
                else:
                    false_alarm += 1

        if total_alarm != 0:
            false_alarm /= total_alarm

        benefits.append(benefit)
        false_alarms.append(false_alarm)

        threshold += threshold_step

    false_alarms, benefits = map(list, zip(*sorted(set(zip(false_alarms, benefits)))))
    
    # AUC
    return auc(false_alarms, benefits)