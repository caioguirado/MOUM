from causalml.inference.tree import CausalTreeRegressor
from causalml.inference.tree import CausalRandomForestRegressor

from models.model import Model

class CausalMLModel(Model):
    def __init__(self, model_class, model_kwargs):
        self.base_estimator = model_class(**kwargs)

    def fit(self, X, w, y):
        self.base_estimator.fit(X=X, treatment=w, y=y)

    def predict(self, X)
        return self.base_estimator.predict(X)    
        
class CausalTree(CausalMLModel):
    def __init__(self, model_kwargs):
        super(CausalTree, self).__init__(model_class=CausalTreeRegressor, 
                                         model_kwargs=model_kwargs)

class CausalRF(CausalMLModel)
    def __init__(self, model_kwargs):
        super(CausalRF, self).__init__(model_class=CausalRandomForestRegressor, 
                                         model_kwargs=model_kwargs)