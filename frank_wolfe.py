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
import os
import pickle
from collections import defaultdict

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--mixture", type=int)
    parser.add_argument("--mixture-two-distance", type=int, default=2)
    parser.add_argument("--debug-N", type=int, default=100)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--config", type=str, default='config.yml')
    
    return parser.parse_args()
args = parse_args()
with open(args.config, "r") as infile:
    config = yaml.full_load(infile)

args_dict = vars(args)
for k, v in args_dict.items():
    config[k] = v

target_folder = args.folder
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Generate sample data
if config['mixture'] == 1:
    means = [1]
    weights = [1]
    covs = [np.eye(1) for _ in range(1)]
    theta0 = 1
elif config['mixture'] == 2:
    a = config['mixture_two_distance']
    means = [-a, a]
    weights = [0.5, 0.5]
    covs = [np.eye(1) for _ in range(2)]
    theta0 = np.random.randn()
else:
    print("Not supported mixture number")
    exit(0)

if 'reweight_type' not in config['experiment']:
    config['experiment']['reweight_type'] = 'all'


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


class NonParametricEstimator:
    def __init__(self, samples, init_theta, reweight_type):
        self.thetas = [init_theta]
        self.alphas = [1]
        self.f = [get_f(init_theta, samples)]
        self.history = []
        self.samples = samples
        self.N = len(samples)
        self.stepwise_plot = False

        self.reweight_type = reweight_type
    
    def get_new_theta(self):
        # Step 1: Find a new direction
        # Find a new mean that maximizes the log likelihood
        restart_init = np.linspace(-5, 5, 100)
        candidate_x0 = [-self.calculate_derivative(x0) for x0 in restart_init]
        x0 = restart_init[np.argmin(candidate_x0)]

        if self.stepwise_plot:
            x = np.linspace(-5, 5, 100)
            y = [self.calculate_derivative(i) for i in x]
            plt.plot(x, y)
            plt.show()

        optim = scipy.optimize.minimize(
                lambda x: -self.calculate_derivative(x), 
                x0,
            )
        theta = float(optim.x)
        optional_print("D:", optim.fun)
        optional_print("Upper bound:", self.N * np.log(1-optim.fun/self.N))
        # theta = np.array([x0])
        return theta, optim.fun

    def calculate_derivative(self, theta):
        new_f = get_f(theta, self.samples)
        denom = np.stack([self.alphas[i] * self.f[i] for i in range(len(self.alphas))]).sum(axis=0)
        return (new_f / denom - 1).sum()

    def add_theta(self, theta):
        new_f = get_f(theta, self.samples)
        self.f.append(new_f)
        self.alphas.append(0)
        self.thetas.append(theta)

    # There can be two strategies
    def reweight_alphas_linear(self):
        def objective(epsilon):
            res = 0
            for i in range(self.N):
                orig = GM_likelihood(self.samples[i], self.alphas[:-1], self.thetas[:-1])
                new = gauss_likelihood(self.samples[i], self.thetas[-1])
                res += np.log((1-epsilon) * orig + epsilon*new)
            return -res
        
        x=np.linspace(0,1,100)
        y=objective(x)

        if self.stepwise_plot:
            plt.plot(x,y)
            plt.title("NLL over epsilon")
            plt.show()

        optim = scipy.optimize.minimize(
            objective,
            0,
            bounds=Bounds(0, 1)
        )
        epsilon = optim.x
        self.alphas *= (1-epsilon)
        self.alphas[-1] = epsilon

        return epsilon
    def reweight_alphas(self):
        if self.reweight_type == 'linear':
            return self.reweight_alphas_linear()

        def objective(alphas):
            res = 0
            for i in range(self.N):
                res += np.log(GM_likelihood(self.samples[i], alphas, self.thetas))
            return -res
        # Define the constraints for the sum of weights
        cons = ({'type': 'eq',
             'fun' : lambda x: x.sum()-1})

        x0 = self.alphas
        bnds = [(0, None) for _ in x0]
        optim2 = scipy.optimize.minimize(
            objective,
            x0,
            bounds=bnds, #  Bounds(0,1)
            method='SLSQP',
            constraints=cons
        )

        # Is this step correct? smaller step?
        # VDM?
        self.alphas = optim2.x.tolist()

        return self.alphas

    def check_new_theta(self, theta, thres=0.01):
        for t in self.thetas:
            if np.abs(theta-t) < thres:
                return False
        
        return True
    
    def evaluate_nnl(self):
        LL = 0
        for k in range(self.N):
            LL += np.log(np.sum(([self.alphas[i] * self.f[i][k] for i in range(len(self.alphas))])))
        return -LL

    def sort(self):
        # # # sort
        pair = list(zip(self.thetas, self.alphas, self.f))
        pair.sort(key=lambda x:x[0])
        self.thetas, self.alphas, self.f= map(list, zip(*pair))
    
    def filter(self, thres=0.01):
        mask = [alpha > thres for alpha in self.alphas]
        self.thetas = [self.thetas[i] for i in range(len(self.thetas)) if mask[i]]
        self.alphas = [self.alphas[i] for i in range(len(self.alphas)) if mask[i]]
    
def merge(thetas, alphas, thres=0.1):

    new_thetas, new_alphas = [], []
    i = 0
    while i < len(alphas):
        j = i + 1
        while j < len(alphas) and \
                thetas[j] - thetas[j-1] <= thres:
                j += 1
        tmp_alphas = alphas[i:j] 
        tmp_thetas = thetas[i:j] 
        
        if sum(tmp_alphas) == 0:
            i = j
            continue
        new_thetas.append(sum([tmp_alphas[i] * tmp_thetas[i] \
                                for i in range(len(tmp_alphas))]) / sum(tmp_alphas))
        new_alphas.append(sum(tmp_alphas))

        i = j
    return new_thetas, new_alphas

def filter(thetas, alphas, thres=0.01):
    mask = [a > thres for a in alphas]
    thetas = [thetas[i] for i in range(len(thetas)) if mask[i]]
    alphas = [alphas[i] for i in range(len(alphas)) if mask[i]]
    return thetas, alphas

    
class Logger:
    def __init__(self, true_thetas, true_alphas):
        self.theta_history = []
        self.alpha_history = []
        self.merged_theta_history = []
        self.merged_alpha_history = []
        self.nnl_history = []
        self.w1_history = []
        self.true_thetas = true_thetas
        self.true_alphas = true_alphas
        self.debug_values = defaultdict(list)
        self.t=0

    def log(self, estimator):
        thetas, alphas = estimator.thetas, estimator.alphas
        self.theta_history.append(thetas)
        self.alpha_history.append(alphas)
        merged_theta, merged_alpha = merge(thetas, alphas)
        self.merged_theta_history.append(merged_theta)
        self.merged_alpha_history.append(merged_alpha)
        self.nnl_history.append(estimator.evaluate_nnl())
        self.w1_history.append(ss.wasserstein_distance(self.true_thetas, thetas,
                                                       self.true_alphas, alphas))
        self.t += 1
    
    def log_key(self, key, value):
        self.debug_values[key].append(value)

    def display(self, thres=0.01):
        formatted_thetas = [f"{theta:.2f}" for theta in self.theta_history[-1]]
        formatted_alphas = [f"{alpha:.2f}" for alpha in self.alpha_history[-1]]
        optional_print("Thetas:", *formatted_thetas)
        optional_print("Alphas:", *formatted_alphas)

        formatted_thetas = [f"{theta:.2f}" for theta in self.merged_theta_history[-1]]
        formatted_alphas = [f"{alpha:.2f}" for alpha in self.merged_alpha_history[-1]]
        optional_print("Merged Thetas:", *formatted_thetas)
        optional_print("Merged Alphas:", *formatted_alphas)

        optional_print("W1", f"{self.w1_history[-1]:.2f}")
        optional_print("NNL", f"{self.nnl_history[-1]:.2f}")
    
    def plot_history(self):
        for i in range(self.t):
            plt.clf()
            plt.xlim(-5, 5)
            plt.ylim(0, 1)
            plt.scatter(self.merged_theta_history[i], self.merged_alpha_history[i])
            plt.title(f"alphas over thetas (t={i})")
            # plt.show()
            plt.pause(10/self.t)
        plt.show()
    
    def plot_all_keys(self):
        for key, value in self.debug_values.items():
            plt.plot(value)
            plt.title(key)
            plt.show()

    def plot_nnl(self):
        plt.plot(self.nnl_history)
        plt.title("NNL over iteration")
        plt.show()
    
    def result(self, filtering_thres):
        res = {}
        res['nnl'] = self.nnl_history[-1]
        res['w1'] = self.w1_history[-1]
        last_theta, last_alpha = self.merged_theta_history[-1], self.merged_alpha_history[-1]
        last_theta, last_alpha = filter(last_theta, last_alpha, filtering_thres)
        res['n'] = len(last_theta)
        return res



def experiment(
        N,
        weights, means, covs,
        stop_thres,
        alpha_thres,
        reweight_type,
        maxiter,
):
    samples = mixture_gaussians(weights, means, covs, N)
    
    if verbose:
        plt.hist(samples, bins=30)
        plt.show()

    estimator = NonParametricEstimator(samples, theta0, reweight_type)
    logger = Logger(means,weights)
    logger.log(estimator)
    # optional_print("True Data NNL", estimator.evaluate_nnl())
    optional_print("True Data NNL", GM_nll(samples, weights, means))


    last_nll = None

    for round in range(maxiter):
        optional_print("="*30)
        optional_print("Round", round)
        optional_print("="*30)

        new_theta, D = estimator.get_new_theta()

        optional_print("New theta:", new_theta)
        estimator.add_theta(new_theta)
        epsilon = estimator.reweight_alphas()
        optional_print("Epsilon:", epsilon)
        estimator.sort()

        logger.log(estimator)
        logger.log_key("derivative", -D)
        
        logger.display()
        nll = logger.nnl_history[-1]
        optional_print("Negative log likelihood", nll)

        if last_nll: optional_print("Change in NLL", last_nll - nll)

        # Step 3: STOP CRITERION
        if last_nll and (last_nll - nll) < stop_thres:
            break
        last_nll = nll

    # ================================================================
    # Post processing
    # ================================================================
    optional_print("="*30)
    optional_print("Post processing")
    optional_print("="*30)
    if verbose:
        logger.plot_history()
        logger.plot_nnl()
        logger.plot_all_keys()
    
    return logger.result(alpha_thres)

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


for N in (range(config['N_start'], config['N_end'], config['N_step'])):
# for n in [2**i for i in range(5, 12)]:
    if args.debug:
        N = args.debug_N
    print("-"*20)
    print("N:", N)
    print("-"*20)
    res_dict = {}

    if args.debug:
        experiment(N, weights, means, covs, **config['experiment'])
        exit(0)

    results = Parallel(n_jobs=num_cores) \
              (delayed(experiment)(N, weights, means, covs, **config['experiment']) \
               for _ in range(config['simulation_times']))

    # for _ in range(config['simulation_times']):
        # res = experiment(n, weights, means, covs, **config['experiment'])
        # for k, v in res.items():
        #     if k not in res_dict:
        #         res_dict[k] = []
        #     res_dict[k].append(v)
    for k, v in results[0].items():
        res_dict[f'{k}_mean'] = np.mean([res[k] for res in results])
        res_dict[f'{k}_std'] = np.std([res[k] for res in results])

    res_dict['N'] = N

    with open(os.path.join(target_folder, f"{N}.pkl"), 'wb') as f:
        pickle.dump(res_dict, f)
    print(res_dict)




# ============================================================
# Plot the sampled data
# ============================================================

# optional_print("Means:", means)
# optional_print("Weights:", weights)
# plt.hist(samples, bins=100)
# plt.title("Samples")
# plt.show()

