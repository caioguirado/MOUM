import numpy as np
from data.tradeoffs.tradeoff import Tradeoff

class LinearTradeoff(Tradeoff):
    
    def __init__(self) -> None:
        pass

    def sigmoid(self, x):
        exp = -12*(x-0.5)
        return 2 / (1 + np.exp(exp)) - 1

    def get_main_effect(self, X):
        mu = 1
        for x in X.T:
            mu *= self.sigmoid(x)
        
        mu_0_0 = 0.5 * mu
        mu_0_1 = -0.5 * mu
        Y_0_0 = mu_0_0 + np.random.normal(0, 1, X.shape[0])
        Y_0_1 = mu_0_1 + np.random.normal(0, 1, X.shape[0])

        return np.concatenate([Y_0_0, Y_0_1], axis=0).T

    def get_tradeoff_effect(self):
        pass

    def create_Y(self, X, n_responses):
        # return Y matrix Yij 
        # where i is the response number 
        # and j the treatment index 
        # (Y_0_0, Y_0_1, ...., Y_N_0, Y_N_1)

        Y = []
        # create main effect
        Y_0 = self.get_main_effect(X)

        # for all other responses, create tradeoffs
        for i in range(1, n_responses):
            self.get_tradeoff_effect()
        pass