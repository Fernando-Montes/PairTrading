import numpy as np
import pandas as pd
import itertools
import math
from src.visualization.visualize import *
from random import seed

class basicStrategy():
    def __init__(self, series = None):
        if series is not None:
            self.series = series.copy()
        else :
            self.series = pd.read_csv('../data/raw/pairs.csv', header=0)
        self.series['hedgedRatio'] = self.series['RATIO']/(1+self.series['RATIO'])

    def _mean_std(self, window):
        '''
        Adds normalized ratio in standard-deviation units
         and normalized hedgeRatio between the stocks to dataframe
        '''
        rolling_mean = self.series['RATIO'].rolling(window).mean()
        rolling_std = self.series['RATIO'].rolling(window).std()
        self.series['RATIO_stdUnits'] = (self.series['RATIO']-rolling_mean)/rolling_std

    def _calculateReturns(self, seriesInput):
        '''
        Returns dataframe with calculated returns
        '''
        series = seriesInput.copy()
        rangeSeries = series.index
        previousPosition = 0
        series.loc[rangeSeries[0],'hedgedRatioStr'] =  series.loc[rangeSeries[0],'hedgedRatio']
        series.loc[rangeSeries[0],'ABC_hedgedCapital'] = (series.loc[rangeSeries[0],'hedgedRatioStr'] *
                                             series.loc[rangeSeries[0],'ABC'] * series.loc[rangeSeries[0],'Position'])
        series.loc[rangeSeries[0],'XYZ_hedgedCapital'] = -((1-series.loc[rangeSeries[0],'hedgedRatioStr']) *
                                             series.loc[rangeSeries[0],'XYZ'] * series.loc[rangeSeries[0],'Position'])
        series.loc[rangeSeries[0],'DailyReturn'] = 0
        for i in range(rangeSeries[1],rangeSeries[-1]+1):
            # Do not change allocation unless there is a new position
            if series.loc[i, 'Position'] != previousPosition :
                series.loc[i,'hedgedRatioStr'] =  series.loc[i,'hedgedRatio']
                previousPosition = series.loc[i, 'Position']
            else:
                series.loc[i,'hedgedRatioStr'] = series.loc[i-1,'hedgedRatioStr']
            # Allocated capital at a given day
            series.loc[i,'ABC_hedgedCapital'] = (series.loc[i,'hedgedRatioStr'] *
                                                 series.loc[i,'ABC'] * series.loc[i,'Position'])
            series.loc[i,'XYZ_hedgedCapital'] = -((1-series.loc[i,'hedgedRatioStr']) *
                                                  series.loc[i,'XYZ'] * series.loc[i,'Position'])
            # Daily profit or loss based on the position of the previous day
            series.loc[i,'DailyProfit'] = ( series.loc[i-1,'ABC_hedgedCapital'] *
                                           (series.loc[i,'ABC']-series.loc[i-1,'ABC'])/series.loc[i-1,'ABC']
                                          + series.loc[i-1,'XYZ_hedgedCapital'] *
                                           (series.loc[i,'XYZ']-series.loc[i-1,'XYZ'])/series.loc[i-1,'XYZ'] )
            if np.abs(series.loc[i-1,'ABC_hedgedCapital']) + np.abs(series.loc[i-1,'ABC_hedgedCapital']) != 0:
                series.loc[i,'DailyReturn'] = (series.loc[i,'DailyProfit'] /
                                               ( np.abs(series.loc[i-1,'ABC_hedgedCapital']) +
                                                 np.abs(series.loc[i-1,'ABC_hedgedCapital']) ))
            else :
                series.loc[i,'DailyReturn'] = 0
        series['CumulativeCompoundedReturns'] = (1+series['DailyReturn']).cumprod()-1
        return series

    def positions(self, window, entry, exit):
        '''
        Adds entry and exit positions to data frame according to:
         Buy ABC if Ratio_std > entry (entry long ABC, short XYZ)
         Sell ABC if Ratio_std <= exit (exit long ABC, short XYZ)
         Buy XYZ if Ratio_std < -entry (entry short ABC, long XYZ)
         Sell XYZ if Ratio_std >= -exit (exit short ABC, long XYZ)
         and
         add returns positions
        '''
        self._mean_std(window)
        long = []
        longPrevious = 0
        short = []
        shortPrevious = 0
        for i in self.series.index:
            if self.series.loc[i,'RATIO_stdUnits'] > entry: # entry long ABC, short XYZ
                long.append(1)
                longPrevious = 1
            elif self.series.loc[i,'RATIO_stdUnits'] <= exit: # exit long ABC, short XYZ
                long.append(0)
                longPrevious = 0
            else:
                long.append(longPrevious)
            if self.series.loc[i,'RATIO_stdUnits'] < -entry: # entry short ABC, long XYZ
                short.append(-1)
                shortPrevious = -1
            elif self.series.loc[i,'RATIO_stdUnits'] >= exit: # exit short ABC, long XYZ
                short.append(0)
                shortPrevious = 0
            else:
                short.append(shortPrevious)
        return np.asarray(long)+np.asarray(short)

    def apply(self, window, entry, exit):
        '''
        Adds entry and exit positions to data frame according to:
         Buy ABC if Ratio_std > entry (entry long ABC, short XYZ)
         Sell ABC if Ratio_std <= exit (exit long ABC, short XYZ)
         Buy XYZ if Ratio_std < -entry (entry short ABC, long XYZ)
         Sell XYZ if Ratio_std >= -exit (exit short ABC, long XYZ)
         and
         add returns after applying strategy
        '''
        self.series['Position'] = self.positions(window, entry, exit)
        self.series = self._calculateReturns(self.series)

    def plot(self):
        plotStrategy(self.series)

    def print(self):
        ''' Useful for debugging '''
        return self.series

    def ar(self):
        ''' Calculates AR of first 70% and last 30% set '''
        apr_70 = 100* (math.pow( (1+self.series.iloc[:int(len(self.series)*0.7), self.series.columns.get_loc('DailyReturn')]).prod(), 256/(len(self.series)*0.7) ) - 1)
        apr_30 =  100* (math.pow( (1+self.series.iloc[int(len(self.series)*0.7):, self.series.columns.get_loc('DailyReturn')]).prod(), 256/(len(self.series)*0.3) ) - 1)
        return apr_70, apr_30

    def ARdistribution(self, returnMetricOnly = False):
        '''
        Calculates AR distribution on test data by MC sampling the distribution of daily returns
         obtained by using the strategy
        '''
        seed(10)
        returns = self.series.iloc[int(len(self.series)*0.7):, self.series.columns.get_loc('DailyReturn')]
        returns = returns.dropna()
        apr = []
        for i in range(10000):
            sampled_returns = np.random.choice(returns, len(returns))
            apr.append( 100* (math.pow( (1+sampled_returns).prod(), 256/(len(self.series)*0.3) ) - 1) )
        if returnMetricOnly == False:
            plotHisto(apr, 'Expected AR [%]', bins = 50)
            print ( 'Expected AR is {0:.2f}% +- {1:.2f}% (1-sigma confidence)'.format( \
                            np.mean(apr), np.std(apr)) )
            return None
        else :
            return np.mean(apr), np.std(apr)

    def optimization(self, window, entry_threshold, exit_threshold, verbose = False):
        ''' Grid search of the best parameters to optimize AR train set '''
        parameters = list(itertools.product(window, entry_threshold, exit_threshold))
        bestARtrain = 0
        results = np.array([], dtype=np.float64)
        for param in parameters:
            self.apply(param[0], param[1], param[2])
            ar_train, ar_test = self.ar()
            results = np.concatenate( (results, [param[0], param[1], param[2], ar_train, ar_test]), axis=None )
            if verbose == True:
                print('Window range: {0} - entry: {1} - exit: {2} - AR train: {3:.2f} - AR test: {4:.2f}'.format(
                    param[0], param[1], param[2], ar_train, ar_test))
            if ar_train > bestARtrain:
                bestARtrain = ar_train
                bestParamTrain = param
        print('Best parameters: window range: {0} - entry: {1} - exit: {2} - AR: {3:.2f}%'.format(
                    bestParamTrain[0], bestParamTrain[1], bestParamTrain[2], bestARtrain))
        self.apply(bestParamTrain[0], bestParamTrain[1], bestParamTrain[2])
        return results.reshape((-1,5))
