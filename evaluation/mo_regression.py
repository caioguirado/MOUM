import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def average_rmse(Y_true, Y_pred, multioutput_weights='uniform_average', scale=True):
    if scale:
        scaler = MinMaxScaler()
        Y_true = scaler.fit_transform(Y_true)
        Y_pred = scaler.transform(Y_pred)
    return mean_squared_error(Y_true, Y_pred, squared=False, multioutput=multioutput_weights)

def mse(Y_true, Y_pred, multioutput_weights='uniform_average', scale=True):
    if scale:
        scaler = MinMaxScaler()
        Y_true = scaler.fit_transform(Y_true)
        Y_pred = scaler.transform(Y_pred)
    return mean_squared_error(Y_true, Y_pred, squared=True, multioutput=multioutput_weights)