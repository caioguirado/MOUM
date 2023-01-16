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
            pass

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

class LinearTradeOffSampler(Sampler):
    def __init__(self, N, m, d):
        super().__init__(N, m, d)
        pass

class NonLinearTradeOffSampler(Sampler):
    def __init__(self, N, m, d):
        super().__init__(N, m, d)
        pass


%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np

def sig2(x):
    exp = -20*((x-1)/3)
    return 1 + (1 / ( 1 + np.exp(exp) ))

def sig(x):
    exp = -12 * (x - 0.5)
    return 2 / ( 1 + np.exp(exp) )

def sig(x, k=5):
    # -1 to 1
    return 2/(1 + np.exp(-k*x)) - 1

def sigmoid_compressed(x, k=5, b=0.5):
    x = x / x.max()
    return 2/(1 + np.exp(-k*(x-b))) - 1

def sig3(x):
    s1 = sig(x) * 0.5
    step_1 = np.maximum(x-0.5, 0)
    s2 = 1 - sig(x) * 0.5
    step_2 =  np.maximum(-1*x-0.5, 0)
    return s1 * step_1 + s2 * step_2
    
def get_mu(X, coef):
    mu = 1
    for x in X.T:
        mu *= sig(x)

    mu2 = 1
    for x in X.T:
        mu2 *= 1-sig(x)
    return coef * (mu+mu2)

def get_mu(X, coef):
    return coef * normal_pdf(X[:, 0]) * normal_pdf(X[:, 1])

cov = 0.3
N=10000
m=2

x = np.linspace(0, 1, 100)

# X is uniform
X = np.random.uniform(0, 1, size=(N, m))

mu_0 = get_mu(X, coef=-0.5)
mu_1 = get_mu(X, coef=0.5)
Y_0 = mu_0 + np.random.normal(0, 1, size=N)
Y_1 = mu_1 + np.random.normal(0, 1, size=N)
tao1 = Y_1 - Y_0
gain1 = tao1/Y_0
plt.scatter(X[:, 0], X[:, 1], c=tao1, alpha=0.5)
plt.show()

def normal_pdf(x, mu=0.5, sigma=0.2):
    return 1+-1*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))

def g(X):
    d = X.shape[1]
    return 1 + 9/(d-1) * X.sum(axis=1)

def h(f1x, gx):
    # scale = abs(f1x.min()) + 1
    # return 1 - np.sqrt(f1x / gx)
    return 1 - (f1x / gx)**2

def f1(X):
    mu_0 = get_mu(X, coef=-0.5)
    mu_1 = get_mu(X, coef=0.5)
    Y_0 = mu_0 + np.random.normal(0, 1, size=N)
    Y_1 = mu_1 + np.random.normal(0, 1, size=N)
    tao1 = Y_1 - Y_0
    return tao1

def f2(X):
    return g(X) * h(f1(X), g(X))

# tao2 = f2(X, tao1)
# mu_0_2 = f2(X, mu_0)
# mu_1_2 = f2(X, mu_1)
# Y_0_2 = mu_0_2 + np.random.normal(0, 1, size=N)
# Y_1_2 = mu_1_2 + np.random.normal(0, 1, size=N)
# tao2 = Y_1_2 - Y_0_2

tao2 = f2(X)
plt.scatter(tao1, tao2)
plt.show()

def plot_effects(X, mu_1, mu_0):
    n_rows = 3
    n_cols = 2
    tao1 = f1(X)
    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(n_rows-1):
        ax1 = axs[i, 0]
        x = X[:, i]
        ax1.scatter(x, mu_1, alpha=0.5, label='1')
        ax1.scatter(x, mu_0, alpha=0.5, label='0')
        # ax1.set_ylim(-2, 2)
        ax1.legend()
        # ax1.relim()

        ax2 = axs[i, 1]
        ax2.scatter(x, tao1, alpha=0.5, label='tao')
        # ax2.set_ylim(-9, 9)
        # ax2.relim()
    
    ax1 = axs[-1, 0]
    im = ax1.hexbin(X[:, 0], X[:, 1], C=tao1, gridsize=25)
    # plt.colorbar(im, cax=ax1)

    tao2 = f2(X)
    ax1 = axs[-1, 1]
    im = ax1.hexbin(X[:, 0], X[:, 1], C=tao2, gridsize=25)    
    # fig.delaxes(axs[2,1])
    plt.show()

plot_effects(X, mu_1, mu_0)
plot_effects(X, mu_1_2, mu_0_2, tao2)

plt.hexbin(X[:, 0], X[:, 1], C=tao2, gridsize=25)