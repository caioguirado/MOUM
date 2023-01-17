import numpy as np
from tradeoffs import TradeoffEnum

class Dataset:
    # prop score, n_X_cols, n_responses, tradeoff type, 
    def __init__(self, 
                    n_rows, 
                    X_dim, 
                    n_responses, 
                    tradeoff_type, 
                    prop_score=0.5):
        self.n_rows = n_rows
        self.X_dim = X_dim
        self.n_responses = n_responses
        self.tradeoff_type = tradeoff_type
        self.prop_score = prop_score

        self.X = np.random.uniform(0, 1, size=(self.n_rows, X_dim))
        self.w = np.random.binomial(1, self.prop_score, size=self.n_rows).reshape(-1, 1)

        self.Y = self.create_Y()

    def create_Y(self):
        tradeoff = TradeoffEnum[self.tradeoff_type].value()
        return tradeoff.create_Y(self.X, self.n_responses)