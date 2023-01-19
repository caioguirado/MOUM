import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.model import Model

class RegressorChain(Model):

    # TODO: add support for sampling all possible permutations if d < 10

    def __init__(self, base_estimator, base_estimator_kwargs):
        self.base_estimator = base_estimator
        self.base_estimator_kwargs = base_estimator_kwargs
        self.models = []
        self.scaler = MinMaxScaler()

    def fit(self, X, w, Y):
        sample_chain = np.random.permutation(Y.shape[1])

        Y = self.scaler.fit_transform(Y)

        prev_ys = None
        for d in sample_chain:
            y_d = Y[:, d]
            if prev_ys is not None:
                X = np.concatenate([X, prev_ys], axis=1)
            model_d = self.base_estimator(**self.base_estimator_kwargs)
            model_d.fit(X, w.reshape(-1), y_d)
            self.models.append(model_d)
            if prev_ys is None:
                prev_ys = y_d.reshape(-1, 1)
            else:
                prev_ys = np.concatenate([prev_ys, y_d.reshape(-1, 1)], axis=1)

    def predict(self, X):
        Y_pred = []
        prev_ys = None
        for i, model in enumerate(self.models):
            if prev_ys is not None:
                X = np.concatenate([X, prev_ys], axis=1)
            y_d_pred = model.predict(X).reshape(-1, 1)
            if prev_ys is None:
                prev_ys = y_d_pred
            else:
                prev_ys = np.concatenate([prev_ys, y_d_pred], axis=1)
            if np.count_nonzero(np.isnan(y_d_pred)) > 0:
                print(f'NAN FOUND: {i}')
            Y_pred.append(y_d_pred)

        return self.scaler.transform(np.concatenate(Y_pred, axis=1))