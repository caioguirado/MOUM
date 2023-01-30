import numpy as np
from scipy.stats import multivariate_normal

from .tradeoff import Tradeoff

class HighlyNonLinearTradeoff(Tradeoff):
    
    def __init__(self) -> None:
        self.mus = []

    def get_main_effect(self, X):
        # create random gaussians
        n_gaussians = 4

        # pick mean points
        centers = np.random.uniform(0.1, 0.9, size=(n_gaussians, X.shape[1])).round(1)
        stds = np.random.uniform(0, 0.1, n_gaussians).round(2)
        orientation = np.random.choice([1, -1], size=n_gaussians)
        mu = np.zeros(X.shape[0])
        for i in range(n_gaussians):
            rv = multivariate_normal(mean=centers[i], cov=np.ones(len(centers[i]))*stds[i])
            mu += rv.pdf(X) * orientation[i]
        
        mu_0_0 =  mu
        mu_0_1 = 2 * mu
        Y_0_0 = (mu_0_0 + np.random.normal(0, 0.2, X.shape[0])).reshape(-1, 1)
        Y_0_1 = (mu_0_1 + np.random.normal(0, 0.2, X.shape[0])).reshape(-1, 1)

        return np.concatenate([Y_0_0, Y_0_1], axis=1)

    def get_tradeoff_effect(self, X):
        return self.get_main_effect(X)

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