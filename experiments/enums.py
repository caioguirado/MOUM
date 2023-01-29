from enum import Enum

from .regression.xdim_model import XdimModel
from .regression.ydim_model import YdimModel
from .regression.tradeoff_model import TDModel
from .regression.method_model import MethodModel
from .regression.propensity_model import PropensityModel
from .regression.multirank_scalar import MultirankScalarization
from .regression.get_data_plots import GetPlots
from .regression.algo_adap import AlgoAdaptation

class XPEnum(Enum):
    MM = MethodModel
    TM = TDModel
    XM = XdimModel
    YM = YdimModel
    PM = PropensityModel
    MS = MultirankScalarization
    PL = GetPlots
    AA = AlgoAdaptation