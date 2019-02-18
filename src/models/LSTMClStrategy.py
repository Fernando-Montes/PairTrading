from src.models.BasicStrategy import *
from src.models.utilities import *
from src.models.LSTMClassification import *

class LSTMClStrategy(basicStrategy):

    def apply(self, threshold = 0.5):
        '''
        Adds best possible entry and exit positions
         and add returns after applying strategy
        '''
        fit_range = 20
        lstmCl = LSTMClassification(n_neurons=5, learning_rate=0.001, fit_range=fit_range)
        lstmCl.restore_model('../models/LSTMClassification-best')
        pos = lstmCl.predict(self.series, threshold)
        for i in range(len(pos)):
            self.series.loc[fit_range-1+i, 'Position'] = pos[i]
        self.series = self._calculateReturns(self.series)
