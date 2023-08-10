import numpy as np
import scipy
from scipy.optimize import LinearConstraint, Bounds
import scipy.stats as ss
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)
rng = np.random.default_rng(42)

"""
Given a list of mean and covariance,
draw from the Mixture of Gaussian distribution
according to the weights
"""
def mixture_gaussians(weights, means, covs, size=1):
    assert len(means) == len(covs) and len(covs) == len(weights)

    assert np.sum(weights) == 1

    mid_ind = np.random.choice(len(weights), size, replace=True, p=weights)

    # This iteration is not efficient
    samples = [rng.normal(means[i], covs[i]) for i in mid_ind]

    return np.array(samples).squeeze()


# TODO: Add covs
# No normalization, add 1/\sqrt(2pi) if want real likelihood

gauss_likelihood = lambda x, mean: np.exp(-1/2*(x-mean)**2)


# Mixing different weights
def GM_likelihood(x, weights, means, covs=None):
    M = len(weights)
    res = 0
    for k in range(M):
        res += weights[k] * gauss_likelihood(x, means[k])
    return res

# Negative log likelihood for Gaussian Mixtures
def GM_nll(X, weights, means, covs=None):
    N = X.shape[0]
    res = 0
    for i in range(N):
        res += -np.log(GM_likelihood(X[i], weights, means, covs))
    return res

def format_float_list(name, float_list):
    try:
        formatted_floats = f"{float_list:.2f}"
        return f"{name}: {formatted_floats}"    
    except:
        formatted_floats = ["{:.4f}".format(a) for a in float_list]
        return f"{name}: {', '.join(formatted_floats)}"
