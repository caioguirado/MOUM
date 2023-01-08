import numpy as np

class Sampler:
    def sample(self):
        N = 1000
        m = 5
        d = 2
        X = np.random.rand(N, m)
        W = np.random.rand(N, 1)
        Y_0 = np.random.rand(N, d)
        Y_1 = np.random.rand(N, d)
        tau = Y_0 - Y_1
        return X, W, Y_0, Y_1, tau