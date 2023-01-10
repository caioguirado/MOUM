import numpy as np

class Sampler:
    def __init__(self, N, m, d):
        self.N = N
        self.m = m 
        self.d = d 

    def sample(self):
        N = 1000
        m = 5
        d = 2
        X = np.random.rand(N, m)
        W = np.random.binomial(n=1, p=0.5, size=N).reshape(-1, 1)
        Y_0 = np.random.rand(N, d)
        Y_1 = np.random.rand(N, d)
        Y_obs = np.where(W, Y_1, Y_0)
        tau = Y_0 - Y_1
        return {"X":X, "W":W, "Y_0":Y_0, "Y_1":Y_1, "Y_obs":Y_obs, "tau":tau}

    def hte_sigmoid(self, X):
        for d in range(self.d):

    def sample_synthetic(self):
        X = np.random.uniform(low=0, high=10, size=(self.N, self.m))
        W = np.random.binomial(n=1, p=0.5, size=self.N).reshape(-1, 1)
        e = np.random.rand(self.N, self.d)
        Y_0 = np.random.rand(self.N, self.d)
        Y_1 = np.random.rand(self.N, self.d)

        # sample x (uniform or normal)
        # sample w
        # sample noise
        # sample y_0s
        # sample taos or calculate according to Athey et. al. 2018
        # calculate y_1s
        # calculate Y_obs

def sig(x):
    return 1 + (1 / ( 1 + np.exp(-20*((x-1)/3)) ))


%load_ext autoreload
%autoreload 2
# %matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(0, 1, 100)

def sig(x):
    exp = -20 * (x - 1/3)
    return 1 + (1 / ( 1 + np.exp(exp) ))

def sig(x):
    exp = -12 * (x - 0.5)
    return 2 / ( 1 + np.exp(exp) )

def get_mu(X, coef):
    mu = 1
    for x in X.T:
        mu *= sig(x)
    return coef * mu

def plot(arg):
    plt.plot(arg)
    plt.show()

def scatter(*arg):
    plt.scatter(*arg)
    plt.show()

cov = 0.3
N=10000
m=2

# X is multivariate normal
# X_bound = 5
# diag = np.random.uniform(X_bound, size=m)
# cov_matrix = np.ones((m, m)) * cov
# np.fill_diagonal(cov_matrix, diag)
# X = np.random.multivariate_normal(np.zeros(m), cov_matrix, size=N)

# X is uniform
X = np.random.uniform(0, 1, size=(N, m))

mu_0 = get_mu(X, coef=-0.5)
mu_1 = get_mu(X, coef=0.5)
Y_0 = mu_0 + np.random.normal(0, 1, size=N)
Y_1 = mu_1 + np.random.normal(0, 1, size=N)
tao1 = Y_1 - Y_0
gain1 = tao1/Y_0
plt.scatter(X[:, 0], X[:, 1], c=Y_1-Y_0, alpha=0.5)
plt.show()

def g(X):
    d = X.shape[1]
    return 1 + 9/(d-1) * X.sum(axis=1)

def h(f1x, gx):
    # scale = abs(f1x.min()) + 1
    # return 1 - np.sqrt(f1x / gx)
    return 1 - (f1x / gx)**2

def f2(X, f1x):
    return g(X) * h(f1x, g(X))

tao2 = f2(X, tao1)

plt.scatter(tao1, tao2)
plt.show()

plt.hexbin(X[:, 0], X[:, 1], C=tao1, gridsize=25)
plt.colorbar()
plt.show()

alpha=0.1
plt.scatter(X[:, 0], Y_0, alpha=alpha, label='Y_0')
plt.scatter(X[:, 0], Y_1, alpha=alpha, label='Y_1')
plt.scatter(X[:, 0], tao1, alpha=alpha, label='tao1')
plt.legend()
plt.show()