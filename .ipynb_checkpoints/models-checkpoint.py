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


class HoltWinters:
    
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    
    # series - initial time series
    # slen - length of a season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    
    """
    
    
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, index = 0, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.index = index
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            if (i+self.slen) >= len(self.series):
                a = len(self.series)-1
            else:
                a = i+self.slen
            sum += float(self.series[a] - self.series[i]) / self.slen
        return sum / self.slen  
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])
                
                self.PredictedDeviation.append(0)
                
                self.UpperBond.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBond.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])
                
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 
                                               + (1-self.gamma)*self.PredictedDeviation[-1])
                     
            self.UpperBond.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])


class HWM:
    def __init__(self, data, season_amount, n_step, alpha=0.3, beta = 0.5, gamma = 0.4):
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_step = n_step
        self.L = season_amount

    def init_trend(self, data):
        '''
        Initializes the starter trend based
        on average of trend averages across seasons.
        '''

        all_trend = 0
        for i in range(self.L):
            if i+self.L >= len(self.data):
                a = len(self.data)-1
            else:
                a = i+self.L
            all_trend += (data[a] - data[i])/self.L
        return all_trend / self.L

    def init_seasonality(self, data):
        '''
        1. Get the average level of the seasons
        2. Divide observation by seasonal mean
        3. Average the numbers from the previous step based on amount of seasons
        '''
        seasonality = {}
        seasonal_mean = []
        n_seasons = int(len(data)/self.L)

        # Get the season means
        for s_i in range(n_seasons):
            seasonal_mean.append(sum(data[self.L*s_i:self.L*s_i+self.L])/self.L)


        # init values
        for i in range(self.L):
            sum_season = []
            for s_i in range(n_seasons):
                dat_compt = data[self.L*s_i+i]
                mean_compt = seasonal_mean[s_i]
                season_compt = dat_compt - mean_compt
                sum_season.append(season_compt)
            sum_season_all = np.sum(sum_season)
            seasonality[i] = sum_season_all/n_seasons
        return seasonality


    def smooth(self):
        pred = []
        seasonality = self.init_seasonality(self.data)
        for i in range(len(self.data) + self.n_step):
            #ruleset for initializing the values
            if i ==0:
                smoothened = self.data[0]
                trend = self.init_trend(self.data)
                pred.append(smoothened)

            #ruleset for forecasting
            elif i>=self.n_step:
                window = i-len(self.data)+1
                pred.append((smoothened + trend * window)
                            + seasonality[i%self.L])

            #adapt the trends, seasonalities, etc.
            else:
                level = self.data[i]
                smoothened_p = smoothened

                smoothened = (self.alpha*(level - seasonality[i%self.L])
                              +(1-self.alpha)*(smoothened_p + trend))

                trend = (self.beta * (smoothened - smoothened_p) +
                         (1-self.beta) * trend)

                seasonality[i%self.L] = (self.gamma*(level - smoothened) +
                                        (1-self.gamma)* seasonality[i%self.L])

                pred.append(smoothened + trend + seasonality[i%self.L])

        return pred
