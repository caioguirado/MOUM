import numpy as np
from models.model import Model

class RegressorChain(Model):

    # TODO: add support for sampling all possible permutations if d < 10

    def __init__(self, base_estimator, base_estimator_kwargs):
        self.base_estimator = base_estimator
        self.base_estimator_kwargs = base_estimator_kwargs
        self.models = []

    def fit(self, X, w, Y):
        sample_chain = np.random.permutation(Y.shape[1])

        prev_ys = []
        for i, d in enumerate(sample_chain):
            y_d = Y[:, [d]]
            if i != 0:
                X = np.concatenate([X, prev_ys], axis=1)
            model_d = self.base_estimator(**self.base_estimator_kwargs)
            model_d.fit(X, w, y_d)
            self.models.append(model_d)
            prev_ys.append(y_d)

    def predict(self, X):
        Y_pred = []
        for model in self.models:
            y_d_pred = model.predict(X)
            Y_pred.append(y_d_pred)

        return np.concatenate(Y_pred, axis=1)