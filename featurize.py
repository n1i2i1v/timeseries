from func import select_ts, tscrossvalid, optimizer
import numpy as np
import pandas as pd
from datetime import datetime

def select_ts_n(index):
    #initiate the conditions
    train = pd.read_csv("ts_train.csv")
    test = pd.read_csv("ts_test.csv")
    train = pd.read_csv("ts_train.csv")
    test = pd.read_csv("ts_test.csv")
    test['ACTUAL'] = 0
    results_hwm = pd.read_csv("prim5.csv")
    res = np.array(results_hwm.value)
    test.ACTUAL = np.array(res)
    train.Date = train.Date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))
    test.Date = test.Date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))
    train['cold'] = train.Temp.apply(lambda x: 10 if x >= 5 else 100)
    train['hot'] = train.Temp.apply(lambda x: 100 if x >= 25 else 10)
    test['cold'] = test.Temp.apply(lambda x: 10 if x >= 5 else 100)
    test['hot'] = test.Temp.apply(lambda x: 100 if x >= 25 else 10)
    for val in (train, test):
        val['year']=val.Date.dt.year 
        val['month']=val.Date.dt.month 
        val['day']=val.Date.dt.day
        
    return (train[train.tsID == index].copy(),
            test[test.tsID == index].copy())

def featurize(i):
            
    #featurizing part
    train, test = select_ts_n(i)
    exog_train = train
    exog_train['sin1'] = np.sin(2 * np.pi * exog_train.ID)
    exog_train['cos1'] = np.cos(2 * np.pi * exog_train.ID)
    exog_train['temp_sin'] = np.sin(2 * np.pi * exog_train.Temp)
    exog_train['temp_cos'] = np.cos(2 * np.pi * exog_train.Temp)
    exog_train['temp3'] = exog_train.Temp.apply(lambda x: x**3)
    exog_train['temp_sin3'] = np.sin(2 * np.pi * exog_train.temp3)
    exog_train['temp_cos3'] = np.cos(2 * np.pi * exog_train.temp3)
    exog_train['temp5'] = exog_train.Temp.apply(lambda x: x**5)
    exog_train['temp_sin5'] = np.sin(2 * np.pi * exog_train.temp5)
    exog_train['temp_cos5'] = np.cos(2 * np.pi * exog_train.temp5)
    exog_train['temp7'] = exog_train.Temp.apply(lambda x: x**7)
    exog_train['temp_sin7'] = np.sin(2 * np.pi * exog_train.temp7)
    exog_train['temp_cos7'] = np.cos(2 * np.pi * exog_train.temp7)
    exog_train['temp9'] = exog_train.Temp.apply(lambda x: x**9)
    exog_train['temp_sin9'] = np.sin(2 * np.pi * exog_train.temp9)
    exog_train['temp_cos9'] = np.cos(2 * np.pi * exog_train.temp9)
    exog_train['res'] = exog_train.ACTUAL.apply(lambda x: x)
    exog_train['res2'] = exog_train.ACTUAL.apply(lambda x: x**2)
    exog_train['ressin'] = np.sin(2 * np.pi * exog_train.ACTUAL)
    exog_train['rescos'] = np.cos(2 * np.pi * exog_train.ACTUAL)

    exog_test = test
    exog_test['sin1'] = np.sin(2 * np.pi * exog_test.Temp)
    exog_test['cos1'] = np.sin(2 * np.pi * exog_test.Temp)
    exog_test['temp_sin'] = np.sin(2 * np.pi * exog_test.Temp)
    exog_test['temp_cos'] = np.cos(2 * np.pi * exog_test.Temp)
    exog_test['temp3'] = exog_test.Temp.apply(lambda x: x**3)
    exog_test['temp_sin3'] = np.sin(2 * np.pi * exog_test.temp3)
    exog_test['temp_cos3'] = np.cos(2 * np.pi * exog_test.temp3)
    exog_test['temp5'] = exog_test.Temp.apply(lambda x: x**5)
    exog_test['temp_sin5'] = np.sin(2 * np.pi * exog_test.temp5)
    exog_test['temp_cos5'] = np.cos(2 * np.pi * exog_test.temp5)
    exog_test['temp7'] = exog_test.Temp.apply(lambda x: x**7)
    exog_test['temp_sin7'] = np.sin(2 * np.pi * exog_test.temp7)
    exog_test['temp_cos7'] = np.cos(2 * np.pi * exog_test.temp7)
    exog_test['temp9'] = exog_test.Temp.apply(lambda x: x**9)
    exog_test['temp_sin9'] = np.sin(2 * np.pi * exog_test.temp9)
    exog_test['temp_cos9'] = np.cos(2 * np.pi * exog_test.temp9)
    exog_test['res'] = exog_test.ACTUAL.apply(lambda x: x)
    exog_test['res2'] = exog_test.ACTUAL.apply(lambda x: x**2)
    exog_test['ressin'] = np.sin(2 * np.pi * exog_test.ACTUAL)
    exog_test['rescos'] = np.cos(2 * np.pi * exog_test.ACTUAL)
    
    train_f = exog_train.drop(['Date', 'ID'],axis = 1).dropna()
    test_f = exog_test.drop(['Date', 'ID'],axis = 1).dropna()
    return (train_f, test_f)

def test_train_split(i):
    train, test = featurize(i)
    #make a huge mess, given old predictions
    merge_train = train.append(test, ignore_index =True)
    m = int(merge_train.shape[0]/5)
    merge_train_f = merge_train[:-m]

    test = merge_train[-300:].dropna().drop(['ACTUAL'], axis=1)
    n = int(merge_train.shape[0]/3)

    x_train = merge_train_f.iloc[:n].dropna().drop(['ACTUAL'], axis=1)
    x_test = merge_train_f.iloc[n:].dropna().drop(['ACTUAL'], axis=1)
    y_train = merge_train_f.ACTUAL.iloc[:n]
    y_test = merge_train_f.ACTUAL.iloc[n:]
    return (x_train, x_test, y_train, y_test, test)