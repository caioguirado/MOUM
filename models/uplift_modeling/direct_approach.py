from causalml.inference.tree import CausalTreeRegressor
from causalml.inference.tree import CausalRandomForestRegressor

from .causal_model import CausalMLModel
        
class CausalTree(CausalMLModel):
    def __init__(self, **model_kwargs):
        super().__init__(model_class=CausalTreeRegressor, 
                                         **model_kwargs)

class CausalRF(CausalMLModel):
    def __init__(self, **model_kwargs):
        super().__init__(model_class=CausalRandomForestRegressor, 
                                        **model_kwargs)