import numpy as np
import scipy
from scipy.optimize import LinearConstraint, Bounds
import scipy.stats as ss
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import yaml
from collections import defaultdict

from utils import *



# Generate sample data
K = 1
D = 1
means = np.random.rand(K, D) * 10
means = [[0]]
weights = np.random.rand(K)
weights /= np.sum(weights)
weights = [1]
covs = [np.eye(D) for _ in range(K)]

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

def optional_print(*msg, **kwargs):
    if verbose: print(*msg, **kwargs)


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

    # One iteration of update
    # Find the argmax of the gradient
    def gradient(theta):
        res = 0
        for i in range(N):
            # This may have overflow or instability problem
            res += gauss_likelihood(samples[i], theta) / \
                GM_likelihood(samples[i], alphas, thetas) # -1
            if np.isnan(res).any():
                optional_print("Has NaN in gradient calculation!")
                exit(0)
        
        return res / N
    optional_print("True Data NNL", GM_nll(samples, weights, means))

    last_likelihood = None

    for round in range(maxiter):
        optional_print("="*30)
        optional_print("Round", round)
        optional_print("="*30)

        optional_print("alphas:", [float(t) for t in alphas])
        optional_print("thetas:", [float(t) for t in thetas])
        optional_print("Negative log likelihood", GM_nll(samples, alphas, thetas))
        
        # Step 1: Find a new direction
        # Find a new mean that maximizes the log likelihood
        candidate_x0 = [-gradient(x0) for x0 in (range(20))]
        x0 = np.argmin(candidate_x0)

        optim = scipy.optimize.minimize(
                lambda x: -gradient(x), 
                x0,
            )
        
        if optim.fun > -0.001:
            optional_print("Gradient small, abort")
            optional_print("Gradient:", optim.fun)
            exit(0)
        theta = optim.x
        

        # Step 2: line search for proper alpha
        def objective2(alpha):
            res = 0
            for i in range(N):
                orig = GM_likelihood(samples[i], alphas, thetas)
                new = gauss_likelihood(samples[i], theta)
                res += np.log((1-alpha) * orig + alpha*new)
            return -res / N

        optim2 = scipy.optimize.minimize(
            objective2,
            0,
            bounds=Bounds(0, 1)
        )
        alpha = optim2.x
        likelihood = optim2.fun


        alphas = np.concatenate([(1-alpha)*alphas, alpha])
        assert np.abs(np.sum(alphas) - 1) < 0.01, \
                "Alphas should sum to one. Now it is %f" % np.sum(alphas)
        alphas /= np.sum(alphas)

        thetas = np.concatenate([thetas, theta])



        # Step 3: STOP CRITERION
        if last_likelihood and np.abs(last_likelihood - likelihood) < stop_thres:
            break
        last_likelihood = likelihood

        

    # ================================================================
    # Post processing
    # ================================================================

    results = [(w, a) for (w, a) in zip(thetas, alphas)]
    optional_print("Original length of results is:", len(results))
    optional_print("After filtering, the length becomes: ", end='')
    results = [(w, a) for (w, a) in results if a > alpha_thres]
    optional_print(len(results))

    w1 = ss.wasserstein_distance(np.array(means).reshape(-1), thetas, u_weights=weights, v_weights=alphas)


    return {
        "number of support": len(results), # Number of support after filtering
        "W1": w1, # Wasserstein 1 distance
    }


with open("config.yml", "r") as infile:
    config = yaml.full_load(infile)



# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="NPMLE",
    
    # track hyperparameters and run metadata
    config=config
)

# define our custom x axis metric
wandb.define_metric("N")
# set all other train/ metrics to use this step
wandb.define_metric("*", step_metric="N")



for n in tqdm(range(1000, 2000, 100)):
    res_dict = {}
    for _ in range(config['simulation_times']):
        res = experiment(n, weights, means, covs, **config['experiment'])
        for k, v in res.items():
            if k not in res_dict:
                res_dict[k] = []
            res_dict[k].append(v)
    for k, v in res_dict.items():
        res_dict[k] = np.mean(v)

    res_dict['N'] = n
    wandb.log(res_dict)
    




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




