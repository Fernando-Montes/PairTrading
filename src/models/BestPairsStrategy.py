from src.models.BasicStrategy import *
from src.models.utilities import *

class bestPairsStrategy(basicStrategy):

    def apply(self):
        '''
        Adds best possible entry and exit positions
         and add returns after applying strategy
        '''
        self.series = bestPairs(self.series)
        self.series = self._calculateReturns(self.series)

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
            series.loc[i,'hedgedRatioStr'] =  series.loc[i,'hedgedRatio']
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

def bestPairs(seriesInput):
    '''
    Adds entry points to data frame according to:
     If ABC moves up and XYZ moves down (entry long ABC, short XYZ)
     If ABC moves down and XYZ moves up (entry short ABC, long XYZ)
     Otherwise do nothing
    '''
    series = seriesInput.copy()
    for t in range(len(series)-1):
        if series.loc[t+1,'ABC'] > series.loc[t,'ABC'] and \
           series.loc[t+1,'XYZ'] < series.loc[t,'XYZ'] :
            series.loc[t, 'Position'] = 1
        elif series.loc[t+1,'ABC'] < series.loc[t,'ABC'] and \
             series.loc[t+1,'XYZ'] > series.loc[t,'XYZ'] :
            series.loc[t, 'Position'] = -1
        else :
            series.loc[t, 'Position'] = 0
    return series
