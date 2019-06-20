import numpy as np
import pandas as pd 
from xgboost import XGBRegressor 
from sklearn.model_selection import TimeSeriesSplit
import warnings                                  
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 9]
from sklearn.preprocessing import StandardScaler
from itertools import product                   
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize         
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
from models import HoltWinters, HWM

def select_ts(index):
    train = pd.read_csv("ts_train.csv")
    test = pd.read_csv("ts_test.csv")
    return (train[train.tsID == index].copy(),
            test[test.tsID == index].copy())

def tscrossvalid(params, data, loss=mean_squared_error, lenght=24):
    """
        Returns error on Cross Validation  

    """
    # errors array
    errors = []
    
    values = data
    alpha, beta, gamma = params
    
    # set the number of folds for cross-validation
    tssp = TimeSeriesSplit(n_splits=3) 
    
    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tssp.split(values):

        model = HoltWinters(series=values[train], slen=lenght, 
                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()
        
        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss(predictions, actual)
        errors.append(error)
        
    return np.mean(np.array(errors))

def optimizer(data):
    # initializing model parameters alpha, beta and gamma
    x = [0, 0, 0]

    # Minimizing the loss function
    opt = minimize(tscrossvalid, x0=x,
                   args=(data, mean_squared_error),
                   method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
                  )

    # Take optimal values...
    alpha_final, beta_final, gamma_final = opt.x
    return(alpha_final, beta_final, gamma_final)
