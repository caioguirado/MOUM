from causalml.inference.meta import BaseTRegressor, BaseSRegressor

from .causal_model import CausalMLModel  
from models.enums import ModelEnum
        
class TLearner(CausalMLModel):
    def __init__(self, **model_kwargs):
        model_kwargs['learner'] = ModelEnum[model_kwargs['learner']].value()
        super().__init__(model_class=BaseTRegressor, 
                                         **model_kwargs)

class SLearner(CausalMLModel):
    def __init__(self, **model_kwargs):
        model_kwargs['learner'] = ModelEnum[model_kwargs['learner']].value()
        super().__init__(model_class=BaseSRegressor, 
                         **model_kwargs)