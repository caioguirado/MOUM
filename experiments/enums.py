from enum import Enum

from .regression.xdim_model import XdimModel
from .regression.ydim_model import YdimModel
from .regression.tradeoff_model import TDModel
from .regression.method_model import MethodModel
from .regression.propensity_model import PropensityModel

class XPEnum(Enum):
    MM = MethodModel
    TM = TDModel
    XM = XdimModel
    YM = YdimModel
    PM = PropensityModel