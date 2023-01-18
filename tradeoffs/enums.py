from enum import Enum
from .linear import LinearTradeoff
from .non_linear import NonLinearTradeoff
from .highly_non_linear import HighlyNonLinearTradeoff

class TradeoffEnum(Enum):
    LINEAR = LinearTradeoff
    NON_LINEAR = NonLinearTradeoff
    HIGHLY_NON_LINEAR = HighlyNonLinearTradeoff