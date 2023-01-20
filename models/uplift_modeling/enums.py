from enum import Enum

from sklearn.ensemble import RandomForestRegressor

class BaseModelEnum(Enum):
    RF = RandomForestRegressor