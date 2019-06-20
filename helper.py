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

train = pd.read_csv("ts_train.csv")
test = pd.read_csv("ts_test.csv")

def select_ts(index):
    return (train[train.tsID == index].copy(),
            test[test.tsID == index].copy())
train1, test1 = select_ts(1)

plt.plot(train1.ACTUAL)
x = 369
ypred1 = np.array(train1.ACTUAL[x:x+300])
plt.plot(train1.ACTUAL[x:x+300])

df_main = pd.Series(ypred1, index = test1.ID)

def selector(i, x = 369, trend = 0, sc = 1):
    trend = 0
    sc = 1
    train2, test2 = select_ts(i)
    n = len(test2)
    ypred2 = (np.array(train2.ACTUAL[x:x+n]) + trend)*sc
    df_maini = (pd.Series(ypred2, index = test2.ID))
    plt.plot(train2.ACTUAL)
    plt.plot((train2.ACTUAL[x:x+300] + trend)*sc)
    return df_maini
def selector2(i, x = 150, trend = 0,  sc = 1):
    trend = 0
    sc = 1
    train12, test12 = select_ts(i)
    n = len(test12)
    ypred12 = (np.array(train12.ACTUAL[x:x+n]) + trend)*sc 
    df_maini = pd.Series(ypred12, index = test12.ID)
    plt.plot(train12.ACTUAL)
    plt.plot((train12.ACTUAL[x:x+300]  + trend)*sc)
    return df_maini

df_main = df_main.append(selector(2, 369))
df_main = df_main.append(selector(3))
df_main = df_main.append(selector(4))
df_main = df_main.append(selector(5))
df_main = df_main.append(selector(6))
df_main = df_main.append(selector(7))
df_main = df_main.append(selector(8))
df_main = df_main.append(selector(9))
df_main = df_main.append(selector(10))
df_main = df_main.append(selector(11))
df_main = df_main.append(selector2(12))
df_main = df_main.append(selector2(13))
df_main = df_main.append(selector2(14))
df_main = df_main.append(selector2(15))
df_main = df_main.append(selector2(16))
df_main = df_main.append(selector2(17))
df_main = df_main.append(selector2(18))
df_main = df_main.append(selector2(19))
df_main = df_main.append(selector2(20))
df_main = df_main.append(selector2(21))
df_main = df_main.append(selector2(22))
df_main.to_csv("submission/helper.csv")
