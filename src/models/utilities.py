import pandas as pd
import numpy as np
from random import sample, seed

def bestPositions(seriesInput):
    '''
    Adds entry points to data frame according to positions that net a profit (looking at the future)
    '''
    series = seriesInput.copy()
    for t in range(len(series)-1):
        if series.loc[t+1,'ABC']*series.loc[t,'RATIO'] - series.loc[t+1,'XYZ'] > 0 :
            series.loc[t, 'IdealPosition'] = 1
        else :
            series.loc[t, 'IdealPosition'] = -1
    return series

def sklearnFormat(seriesInput, fit_range, return_y = True):
        '''
        Formats input data into sklearn input data
        '''
        if return_y == True:
            series = seriesInput.copy()
            position = np.array(series['IdealPosition'])
            series = series[['RATIO', 'DIFF', 'ABC', 'XYZ']]
            X = []
            y = []
            for t in range(len(series)-fit_range):
                X.append( np.array(series[t:(t+fit_range)]) )
                y.append ( position[t+fit_range-1] )
            return np.array(X).reshape(-1, fit_range*4), np.array(y)
        else :
            series = seriesInput.copy()
            series = series[['RATIO', 'DIFF', 'ABC', 'XYZ']]
            X = []
            for t in range(len(series)-fit_range+1):
                X.append( np.array(series[t:(t+fit_range)]) )
            return np.array(X).reshape(-1, fit_range*4)

def formatData(seriesInput, fit_range):
    '''
    Formats input data into RNN input data
    '''
    series = seriesInput.copy()
    series = series[['RATIO', 'DIFF', 'ABC', 'XYZ']]
    X_total = []
    for t in range(len(series)-fit_range+1):
         X_total.append( np.array(series[t:(t+fit_range)]) )
    return np.array(X_total)
