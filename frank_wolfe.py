import numpy as np
import scipy
from scipy.optimize import LinearConstraint, Bounds
import scipy.stats as ss
import matplotlib.pyplot as plt
from tqdm import tqdm
# import wandb
import yaml
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing

from utils import *

import argparse

with open("config.yml", "r") as infile:
    config = yaml.full_load(infile)

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--mixture", type=int)
    parser.add_argument("--mixture-two-distance", type=int, default=2)
    parser.add_argument("--debug-N", type=int, default=100)
    parser.add_argument("--debug", action='store_true')
    
    return parser.parse_args()
args = parse_args()
args_dict = vars(args)
for k, v in args_dict.items():
    config[k] = v

# Generate sample data

if config['mixture'] == 1:
    means = [[1]]
    weights = [1]
    covs = [np.eye(1) for _ in range(1)]
elif config['mixture'] == 2:
    a = config['mixture_two_distance']
    means = [[a], [-a]]
    weights = [0.5, 0.5]
    covs = [np.eye(1) for _ in range(2)]
else:
    print("Not supported mixture number")
    exit(0)

print("-"*30)
print("Config")
print("-"*30)
print(config)

# Number of supports
# Remove too small alphas
# Wasserstein distance to zero
    # sum of alpha_i * |\theta_i|
# Increase N, observe
    # The number of support -> 1
    # How does the W distance grow
        # COMPARE with
        # suppose we know it's one sample
        # 1 / sqrt(N)
        # |theta_i|
    # run m experiments for this

# Theratical bound
# NPMLE 1/ sqrt(N) (lower bound) , 1/ N^0.25 (theoratical upper bound).

# Two mixtures
# ==
# two symmetric mixtures (-theta, theta) (0.5, 0.5)
# same experiment as before
# fix N, increase theta, at which point, when can algo tell there are two components (more far apart) => The threshold

# Wasserstein distance
    # linear program
    # 
    # as N grows
# theoratical bound
    # 1/N^(1/8), 1/N^(1/6), tightness

# break the symmetry a little bit. un-balanced weights
# plot the mixtures

verbose=False 
if args.debug:
    verbose = True

def optional_print(*msg, **kwargs):
    if verbose: print(*msg, **kwargs)


def get_f(theta, samples):
    f = gauss_likelihood(samples, theta)
    return f

def experiment(
        N,
        weights, means, covs,
        stop_thres,
        alpha_thres,
        maxiter,

):
    samples = mixture_gaussians(weights, means, covs, N)

    # Initialize with MLE estimator
    thetas = [(np.mean(samples))]
    alphas = np.array([1])
    # thetas = [-1, 1]
    # alphas = np.ones(2) * 0.5
    f_thetas = [get_f(thetas[0], samples)]
    # f_thetas = [get_f(thetas[0], samples), get_f(thetas[1], samples)]

    # TODO: NPMLE local minimum
    # new direction: precision not enough
    # increase resolution

    # Different initializations
    # Two mixture, wrong bias
    # Grid resolution for new 
    # gradient descent: stop criterion


    # One iteration of update
    # Find the argmax of the gradient
    def gradient(theta):
        new_f = get_f(theta, samples)
        denom = np.stack([alphas[i] * f_thetas[i] for i in range(len(alphas))]).sum(axis=0)
        return (new_f / denom).mean()
    optional_print("True Data NNL", GM_nll(samples, weights, means))

    last_likelihood = None

    for round in range(maxiter):
        optional_print("="*30)
        optional_print("Round", round)
        optional_print("="*30)

        optional_print(format_float_list("thetas", thetas))
        optional_print(format_float_list("alphas", alphas))
        optional_print("Negative log likelihood", GM_nll(samples, alphas, thetas))
        
        # Step 1: Find a new direction
        # Find a new mean that maximizes the log likelihood
        restart_init = np.linspace(-5, 5, 100)
        candidate_x0 = [-gradient(x0) for x0 in restart_init]
        x0 = restart_init[np.argmin(candidate_x0)]

        # x = np.linspace(-5, 5, 100)
        # y = [gradient(i) for i in x]
        # plt.plot(x, y)
        # plt.show()
        # exit(0)

        optim = scipy.optimize.minimize(
                lambda x: -gradient(x), 
                x0,
            )
        theta = optim.x
        # theta = np.array([x0])
        

        # Step 2: line search for proper alpha
        # def objective2(alpha):
        #     res = 0
        #     for i in range(N):
        #         orig = GM_likelihood(samples[i], alphas, thetas)
        #         new = gauss_likelihood(samples[i], theta)
        #         res += np.log((1-alpha) * orig + alpha*new)
        #     return -res / N

        # optim2 = scipy.optimize.minimize(
        #     objective2,
        #     0,
        #     bounds=Bounds(0, 1)
        # )

        def objective2(new_alphas):
            res = 0
            new_thetas = np.concatenate([thetas, theta])
            for i in range(N):
                res += np.log(GM_likelihood(samples[i], new_alphas, new_thetas))
            return -res / N

        # Define the constraints for the sum of weights
        cons = ({'type': 'eq',
             'fun' : lambda x: x.sum()-1})
        x0 = np.concatenate([alphas, [0]])
        bnds = [(0, None) for _ in x0]
        optim2 = scipy.optimize.minimize(
            objective2,
            x0,
            bounds=bnds, #  Bounds(0,1)
            method='SLSQP',
            constraints=cons
        )

        alphas = optim2.x
        likelihood = optim2.fun # TODO: LOG


        # alphas = np.concatenate([(1-alpha)*alphas, alpha])
        assert np.abs(np.sum(alphas) - 1) < 0.01, \
                "Alphas should sum to one. Now it is %f" % np.sum(alphas)
        alphas /= np.sum(alphas)

        thetas = np.concatenate([thetas, theta])
        f_thetas.append(get_f(theta, samples))

        # # # sort
        pair = list(zip(thetas, alphas, f_thetas))
        pair.sort(key=lambda x:x[0])
        # # if round ==1:
        # #     from IPython import embed
        # #     embed() or exit(0)
        thetas, alphas, f_thetas = zip(*pair)
        f_thetas = list(f_thetas)

        if last_likelihood: optional_print("llh", last_likelihood - likelihood)

        w1 = ss.wasserstein_distance(np.array(means).reshape(-1), thetas, u_weights=weights, v_weights=alphas)
        optional_print("W1", f"{w1:.2f}")

        # Step 3: STOP CRITERION
        if last_likelihood and np.abs(last_likelihood - likelihood) < stop_thres:
            break
        last_likelihood = likelihood

    # ================================================================
    # Post processing
    # ================================================================
    optional_print("="*30)
    optional_print("Post processing")
    optional_print("="*30)

    results = [(w, a) for (w, a) in zip(thetas, alphas)]
    optional_print("Original length of results is:", len(results))
    optional_print("After filtering, the length becomes: ", end='')
    results = [(w, a) for (w, a) in results if a > alpha_thres]
    optional_print(len(results))


    w1 = ss.wasserstein_distance(np.array(means).reshape(-1), thetas, u_weights=weights, v_weights=alphas)

    optional_print(format_float_list("w1", w1))
    optional_print(format_float_list("thetas", thetas))
    optional_print(format_float_list("alphas", alphas))

    return {
        "number of support": len(results), # Number of support after filtering
        "W1": w1, # Wasserstein 1 distance
    }


num_cores = multiprocessing.cpu_count()
if args.debug:
    num_cores = 1


for n in (range(config['N_start'], config['N_end'], config['N_step'])):
# for n in [2**i for i in range(5, 12)]:
    if args.debug:
        n = args.debug_N
    print("-"*20)
    print("N:", n)
    print("-"*20)
    res_dict = {}

    if args.debug:
        experiment(n, weights, means, covs, **config['experiment'])
        exit(0)

    results = Parallel(n_jobs=num_cores) \
              (delayed(experiment)(n, weights, means, covs, **config['experiment']) \
               for _ in range(config['simulation_times']))

    # for _ in range(config['simulation_times']):
        # res = experiment(n, weights, means, covs, **config['experiment'])
        # for k, v in res.items():
        #     if k not in res_dict:
        #         res_dict[k] = []
        #     res_dict[k].append(v)
    for k, v in results[0].items():
        res_dict[k] = np.mean([res[k] for res in results])
    

    res_dict['N'] = n
    # wandb.log(res_dict)
    print(res_dict, flush=True)






# ============================================================
# Plot the sampled data
# ============================================================

# optional_print("Means:", means)
# optional_print("Weights:", weights)
# plt.hist(samples, bins=100)
# plt.title("Samples")
# plt.show()





# optional_print("Adding new theta Data likelihood", data_GM_likelihood(samples, alphas, thetas))
exit(0)

# === 



# def f(theta):
#     res = 0
#     for i in range(100):
#         # res += np.exp(-1/2*(samples[i] - theta)**2)
#         res += gauss_likelihood(samples[i], theta)
#     return res/100

X = np.linspace(-2, 2, 40)
y = [objective(x) for x in X]
plt.plot(X, y)
plt.show()


exit(0)



"""
Assumption:
    1. There is only one customer
"""

# The number of products
n = 1

# The number of features
feature = 1

# Total number of purchasing history
T = 20

# Sales for product at different time | n x T
N = np.random.randint(1, feature, (n, T))



Nj / N, 

# Features of the products | n x T x feature
z = np.random.rand(n, T, feature)


# The distribution of the customer
def generate_customer_mixture_guassian(m):
    means = []
    covs = []
    for _ in range(m):
        means.append(np.random.rand(feature) * 0)
        covs.append(np.diag(np.random.randint(1, 1, feature)))
    weights = np.random.rand(m)
    weights /= np.linalg.norm(weights, 1)
    weights = weights.tolist()

    return lambda : mixture_gaussians(means, covs, weights)

# The customer mixture
Q = generate_customer_mixture_guassian(1)

## Start with naive one modal or two modals.


# The sampled omega for different t | T x feature
true_omega = [Q() for _ in range(T)]

# The z is fixed
ff = lambda x:f(x, z)

# Randomly initialize g_0 | length n x T
g0 = ff(np.random.randn(feature))

# g_true = np.stack([ff(omega) for omega in true_omega], axis=-1)

# optional_print(g_true.shape)
# from IPython import embed
# embed() or exit(0)



# Algorithm start

# Step 1: support finding

# gs has the shape n x T
# 
def grad_nll(omega, N, g,):
    ans = 0
    new_f = ff(omega)
    for t in range(T):
        for j in range(n):
            ans += N[j][t] / g[j][t] * new_f[j][t]
    return -ans / np.sum(N)

# omega feature x 1
def nll(g, N):
    ans = 0
    for t in range(T):
        for j in range(n):
            ans += N[j][t] * np.log(g[j][t])
    return -ans / (np.sum(N))

def grad_sq(omega, N, g):
    ans = 0
    new_f = ff(omega)
    for t in range(T):
        Nt = np.sum([N[i][t] for i in range(len(N))])
        for j in range(n):
            ans += (Nt * g[j][t] - N[j][t]) * new_f[j][t]
    return ans / np.sum(N)

def sq(g, N):
    ans = 0
    for t in range(T):
        Nt = np.sum([N[i][t] for i in range(len(N))])
        for j in range(n):
            yjt = N[j][t] / Nt
            ans += (g[j][t] - yjt)**2
    return ans / (2*np.sum(N))

def grad_l1(omega, N, g):
    ans = 0
    new_f = ff(omega)
    for t in range(T):
        Nt = np.sum([N[i][t] for i in range(len(N))])
        for j in range(n):
            if g[j][t] > N[j][t] / Nt:
                sign = 1
            else:
                sign = -1
            ans += Nt * sign * new_f[j][t]
    return ans / np.sum(N)

def l1(g, N):
    ans = 0
    for t in range(T):
        Nt = np.sum([N[i][t] for i in range(len(N))])
        for j in range(n):
            yjt = N[j][t] / Nt
            ans += np.sum(np.abs(g[j][t] - yjt))
    return ans / (np.sum(N))

list_g = [g0]

# Scalar 1. the dimension of the gaussian
# weighted sum of list_g

g = g0
alpha = 0.001

def get_combined_g(list_g, alpha):
    assert len(list_g) == len(alpha)
    return np.sum(np.stack([alpha[i] * list_g[i] for i in range(len(list_g))]), axis=0)

for _ in range(20):
    objective = lambda x: grad_nll(x, N, g)
    omega_0 = np.random.randn(feature)
    new_omega = scipy.optimize.minimize(objective, omega_0, method='BFGS').x
    new_f = f(new_omega, z)

    list_g.append(new_f)

    def objective(alpha):
        iterate_g = get_combined_g(list_g, alpha)
        return nll(iterate_g, N)
    
    

    alpha0 = np.random.rand(len(list_g))
    alpha0 /= np.sum(alpha0)
    constraint = LinearConstraint(np.ones_like(alpha0), 1, 1)
    bounds = Bounds(np.zeros_like(alpha0))
    alpha = scipy.optimize.minimize(objective, alpha0, constraints=[constraint], bounds=bounds).x

    g = get_combined_g(list_g, alpha)

    # optional_print("NLLL")

    # g = (1-alpha) * g + alpha * new_f
    optional_print("NLL:", nll(g, N))
    optional_print("SQ:", sq(g, N))
    optional_print("l1:", l1(g, N))
    optional_print("g:", g)
    optional_print("len_list_g:", len(list_g))
    # break

# optional_print("NLL for g0:", nll(g0, N, z))

# optional_print("NLL for ", nll((1-alpha)*g0 + alpha*new_f, N, z))




