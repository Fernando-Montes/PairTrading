from src.models.BasicStrategy import *
from src.models.utilities import *
from src.models.RNNClassification import *

class RNNClStrategy(basicStrategy):

    def apply(self, threshold = 0.5):
        '''
        Adds best possible entry and exit positions
         and add returns after applying strategy
        '''
        fit_range = 20
        rnnCl = RNNClassification(n_neurons=20, learning_rate=0.001, fit_range=fit_range)
        rnnCl.restore_model('../models/RNNClassification-best')
        pos = rnnCl.predict(self.series, threshold)
        for i in range(len(pos)):
            self.series.loc[fit_range-1+i, 'Position'] = pos[i]
        self.series = self._calculateReturns(self.series)
