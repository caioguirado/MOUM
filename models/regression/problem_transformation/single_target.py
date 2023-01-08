import numpy as np
from models import Model

class SingleTarget(Model):
    
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.models = []

    def fit(self, X, w, Y):
        for d in Y.shape[1]:
            y_d = Y[:, d].reshape(-1, 1)
            model_d = self.base_estimator.fit(X=X, w=w, y=y_d)
            self.models.append(model_d)

    def predict(self, X):
        Y = []
        for model in self.models:
            y_d_pred = model.predict(X)
            Y.append(y_d_pred)

        return np.concatenate([Y], axis=1)