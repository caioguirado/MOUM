from enum import Enum

from .uplift_modeling.direct_approach import CausalTree, CausalRF
from .uplift_modeling.two_model_approach import SLearner, TLearner
from .regression.problem_transformation import MultiTarget, SingleTarget, RegressorChain

class ModelEnum(Enum):
    CAUSAL_TREE = CausalTree
    CAUSAL_FOREST = CausalRF
    TL = TLearner
    SL = SLearner

class MethodEnum(Enum):
    ST = SingleTarget
    MT = MultiTarget
    RC = RegressorChain