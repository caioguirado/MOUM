import numpy as np
from models.model import Model

class SingleTarget(Model):
    
    # TODO: check for .predict() need to post process with .reshape(-1, 1)

    def __init__(self, base_estimator, base_estimator_kwargs):
        self.base_estimator = base_estimator
        self.base_estimator_kwargs = base_estimator_kwargs
        self.models = []

    def fit(self, X, w, Y):
        for d in range(Y.shape[1]):
            y_d = Y[:, d]
            model_d = self.base_estimator(**self.base_estimator_kwargs)
            model_d.fit(X=X, w=w.reshape(-1), y=y_d)
            self.models.append(model_d)

    def predict(self, X):
        Y_pred = []
        for model in self.models:
            y_d_pred = model.predict(X).reshape(-1, 1)
            Y_pred.append(y_d_pred)

        return np.concatenate(Y_pred, axis=1)