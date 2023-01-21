import numpy as np
from ...model import Model
from .single_target import SingleTarget

class MultiTarget(Model):

    def __init__(self, base_estimator, base_estimator_kwargs):
        self.base_estimator = base_estimator
        self.base_estimator_kwargs = base_estimator_kwargs
        self.first_stage_model = SingleTarget(base_estimator=self.base_estimator, 
                                                base_estimator_kwargs=self.base_estimator_kwargs)
        self.second_stage_model = SingleTarget(base_estimator=self.base_estimator, 
                                                base_estimator_kwargs=self.base_estimator_kwargs)  

    def fit(self, X, w, Y):
        self.first_stage_model.fit(X, w, Y)
        Y_pred = self.first_stage_model.predict(X)
        X = np.concatenate([X, Y_pred], axis=1)
        self.second_stage_model.fit(X, w, Y)

    def predict(self, X):
        Y_pred = self.first_stage_model.predict(X)
        X = np.concatenate([X, Y_pred], axis=1)
        Y_pred = self.second_stage_model.predict(X)
        return Y_pred 