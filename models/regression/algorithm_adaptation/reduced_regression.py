import numpy as np

from models.model import Model

class ReducedRankRegression(Model):

    def __init__(self, rank):
        self.rank = rank

    def fit(self, X, w, Y):
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