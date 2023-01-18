from causalml.inference.meta import BaseTRegressor, BaseSRegressor

from .causal_model import CausalMLModel  
        
class TLearner(CausalMLModel):
    def __init__(self, **model_kwargs):
        super(TLearner, self).__init__(model_class=BaseTRegressor, 
                                         **model_kwargs)

class SLearner(CausalMLModel):
    def __init__(self, model_kwargs):
        super(SLearner, self).__init__(model_class=BaseSRegressor, 
                                         model_kwargs=model_kwargs)