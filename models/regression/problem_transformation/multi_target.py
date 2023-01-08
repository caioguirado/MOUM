import numpy as np
from models.model import Model

class MultiTarget(Model):

    def __init__(self, base_estimator):
        self.first_base_estimator = base_estimator
        self.second_base_estimator = self.first_base_estimator.copy() # check deepcopy
        self.first_stage_models = []
        self.second_stage_models = []

    def fit(self, X, w, Y):
        for d in Y.shape[1]:
            y_d = Y[:, d].reshape(-1, 1)
            model_d = self.first_base_estimator.fit(X=X, w=w, y=y_d)
            self.first_stage_models.append(model_d)

        Y_pred = []
        for model in self.first_stage_models:
            y_d_pred = model.predict(X)  # reshape?
            Y_pred.append(y_d_pred)

        Y_pred = np.concatenate([Y_pred], axis=1)
        X = np.concatenate([X, Y_pred], axis=1)

        for d in Y.shape[1]:
            y_d = Y[:, d].reshape(-1, 1)
            model_d = self.second_base_estimator.fit(X=X, w=w, y=y_d)
            self.second_stage_models.append(model_d)

    def predict(self, X):
        Y_pred = []
        for model in self.first_stage_models:
            y_d_pred = model.predict(X) # reshape?
            Y_pred.append(y_d_pred)

        Y_pred = np.concatenate([Y_pred], axis=1)
        X = np.concatenate([X, Y_pred], axis=1)

        Y_pred = []
        for model in self.second_stage_models:
            y_d_pred = model.predict(X)
            Y_pred.append(Y_pred)

        return np.concatenate([Y_pred], axis=1)