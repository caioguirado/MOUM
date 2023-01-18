from models.model import Model

class CausalMLModel(Model):
    def __init__(self, model_class, **model_kwargs):
        self.base_estimator = model_class(**model_kwargs)

    def fit(self, X, w, y):
        self.base_estimator.fit(X=X, treatment=w, y=y)

    def predict(self, X):
        return self.base_estimator.predict(X)   