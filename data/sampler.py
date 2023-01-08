import numpy as np

class Sampler:
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