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


%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(0, 1, 100)

def sig(x):
    return 1 + (1 / ( 1 + np.exp(-20*((x-1)/3)) ))

plt.plot(x)