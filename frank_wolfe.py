import numpy as np
import scipy
from scipy.optimize import LinearConstraint, Bounds
import scipy.stats as ss
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)
rng = np.random.default_rng(42)




# Given a list of mean and covariance,
# draw from the Mixture of Gaussian distribution
# according to the weights



def mixture_gaussians(weights, means, covs, size=1):
    assert len(means) == len(covs) and len(covs) == len(weights)

    assert np.sum(weights) == 1

    mid_ind = np.random.choice(len(weights), size, replace=True, p=weights)

    # This iteration is not efficient
    samples = [rng.multivariate_normal(means[i], covs[i]) for i in mid_ind]

    return np.array(samples).squeeze()


# TODO: Add covs
# No normalization, add 1/\sqrt(2pi) if want real likelihood

gauss_likelihood = lambda x, mean: np.exp(-1/2*(x-mean)**2)


def GM_likelihood(x, weights, means, covs=None):
    M = len(weights)
    res = 0
    for k in range(M):
        res += weights[k] * gauss_likelihood(x, means[k])
    return res

def GM_nll(X, weights, means, covs=None):
    N = X.shape[0]
    res = 0
    for i in range(N):
        res += -np.log(GM_likelihood(X[i], weights, means, covs))
    return res / N


# Generate Ground Truth
K = 2
D = 1
means = np.random.rand(K, D) * 10
weights = np.random.rand(K)
weights /= np.sum(weights)
covs = [np.eye(D) for _ in range(K)]


# Generate Data
N = 10000

samples = mixture_gaussians(weights, means, covs, N)

# ============================================================
# Plot the sampled data
# ============================================================

print("Means:", means)
print("Weights:", weights)
# plt.hist(samples, bins=100)

# plt.show()



# Initialize with MLE estimator
thetas = [(np.mean(samples))]
alphas = np.array([1])


# One iteration of update

# Find the argmax of the gradient
# 
def gradient(theta):
    res = 0
    for i in range(N):
        # This may have overflow or instability problem
        res += gauss_likelihood(samples[i], theta) / \
            GM_likelihood(samples[i], alphas, thetas) # -1
        if np.isnan(res).any():
            print("Has NaN in gradient calculation!")
            exit(0)
    
    return res / N
print("True Data NNL", GM_nll(samples, weights, means))

# X = np.linspace(0, 20, 20)
# plt.plot([-gradient(x) for x in X])
# plt.show()
# exit(0)

for round in range(100):
    print("="*30)
    print("Round", round)
    print("="*30)

    print("alphas:", [float(t) for t in alphas])
    print("thetas:", [float(t) for t in thetas])
    print("Negative log likelihood", GM_nll(samples, alphas, thetas))
    
    # Step 1: Find a new direction
    # Find a new mean that maximizes the log likelihood
    candidate_x0 = [-gradient(x0) for x0 in (range(20))]
    x0 = np.argmin(candidate_x0)
    optim = scipy.optimize.minimize(
            lambda x: -gradient(x), 
            x0,
        )
    if optim.fun > -0.001:
        print("Gradient small, abort")
        print("Gradient:", optim.fun)
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

    alpha = scipy.optimize.minimize(
        objective2,
        0.5,
        bounds=Bounds(0, 1)
    ).x

    if alpha < 0.01:
        print("Alpha is small, abort")
        print("Alpha:", alpha)
        exit(0)


    alphas = np.concatenate([(1-alpha)*alphas, alpha])
    assert np.abs(np.sum(alphas) - 1) < 0.01, \
            "Alphas should sum to one. Now it is %f" % np.sum(alphas)
    alphas /= np.sum(alphas)

    thetas = np.concatenate([thetas, theta])
    

# print("Adding new theta Data likelihood", data_GM_likelihood(samples, alphas, thetas))
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

# print(g_true.shape)
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

    # print("NLLL")

    # g = (1-alpha) * g + alpha * new_f
    print("NLL:", nll(g, N))
    print("SQ:", sq(g, N))
    print("l1:", l1(g, N))
    print("g:", g)
    print("len_list_g:", len(list_g))
    # break

# print("NLL for g0:", nll(g0, N, z))

# print("NLL for ", nll((1-alpha)*g0 + alpha*new_f, N, z))




