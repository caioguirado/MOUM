from causalml.inference.meta import BaseTRegressor, BaseSRegressor

from .causal_model import CausalMLModel  
from .enums import BaseModelEnum
        
class TLearner(CausalMLModel):
    def __init__(self, **model_kwargs):
        random_state = model_kwargs.pop('random_state')
        model_kwargs['learner'] = BaseModelEnum[model_kwargs['learner']].value(random_state=random_state)
        super().__init__(model_class=BaseTRegressor, 
                                         **model_kwargs)

class SLearner(CausalMLModel):
    def __init__(self, **model_kwargs):
        random_state = model_kwargs.pop('random_state')
        model_kwargs['learner'] = BaseModelEnum[model_kwargs['learner']].value(random_state=random_state)
        super().__init__(model_class=BaseSRegressor, 
                         **model_kwargs)