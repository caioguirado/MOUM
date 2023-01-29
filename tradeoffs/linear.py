import numpy as np
from .tradeoff import Tradeoff

class LinearTradeoff(Tradeoff):
    
    def __init__(self) -> None:
        self.mus = []

    def linear(self, x):
        return x

    def get_main_effect(self, X):
        mu = 1
        for x in X[:, :2].T:
            mu *= self.linear(x)
        
        mu_0_0 = -mu + 0.7
        mu_0_1 = mu - 0.7
        self.mus += [mu_0_0, mu_0_1]
        Y_0_0 = (mu_0_0 + np.random.normal(0, 0.1, X.shape[0])).reshape(-1, 1)
        Y_0_1 = (mu_0_1 + np.random.normal(0, 0.1, X.shape[0])).reshape(-1, 1)

        return np.concatenate([Y_0_0, Y_0_1], axis=1)

    def get_tradeoff_effect(self, X):
        mu = 1
        for x in X.T:
            mu *= 1-self.linear(x)
        
        mu_d_0 = mu - 0.7
        mu_d_1 = -mu + 0.7
        Y_d_0 = (mu_d_0 + np.random.normal(0, 0.1, X.shape[0])).reshape(-1, 1)
        Y_d_1 = (mu_d_1 + np.random.normal(0, 0.1, X.shape[0])).reshape(-1, 1)

        return np.concatenate([Y_d_0, Y_d_1], axis=1)

    def create_Y(self, X, n_responses):
        # return Y matrix Yij 
        # where i is the response number 
        # and j the treatment index 
        # (Y_0_0, Y_0_1, ...., Y_N_0, Y_N_1)

        Y = []
        # create main effect
        Y_0 = self.get_main_effect(X)
        Y.append(Y_0)

        # for all other responses, create tradeoffs
        for i in range(1, n_responses):
            Y_i = self.get_tradeoff_effect(X)
            Y.append(Y_i)
        
        return np.concatenate(Y, axis=1)