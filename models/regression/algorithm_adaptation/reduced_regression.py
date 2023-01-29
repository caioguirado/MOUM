import numpy as np

from ...model import Model

class RRRBase(Model):

    def __init__(self, rank=1):
        self.rank = rank

    def fit(self, X, Y):
        XX = np.dot(X.T, X)
        XY = np.dot(X.T, Y)
        X_inv = np.linalg.pinv(XX)
        X_inv_Y = np.dot(X_inv, XY)
        B = np.dot(XY.T, X_inv_Y)
        _U, _S, V = np.linalg.svd(B)
        self.W = V[0:self.rank, :].T
        self.A = np.dot(X_inv, np.dot(XY, self.W)).T

    def predict(self, X):
        return np.dot(X, np.dot(self.A.T, self.W.T))

class ReducedRankRegression(Model):

    def __init__(self, rank=1):
        self.rank = rank
        self.treatment_models = []

    def fit(self, X, w, Y):
        for t in [0, 1]:
            mask = (w == t).flatten()
            X_filtered = X[mask]
            Y_filtered = Y[mask]
            model = RRRBase()
            model.fit(X_filtered, Y_filtered)
            self.treatment_models.append(model)

    def predict(self, X):
        preds = []
        for model in self.treatment_models:
            pred = model.predict(X)
            preds.append(pred)

        return preds[1] - preds[0]