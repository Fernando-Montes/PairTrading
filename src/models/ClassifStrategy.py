from src.models.BasicStrategy import *
from src.models.utilities import *
from src.models.Classification import *

class classifStrategy(basicStrategy):

    def apply(self, model):
        '''
        Adds best possible entry and exit positions
         and add returns after applying strategy
        '''

        fit_range = 10
        seriesFull = pd.read_csv('../data/raw/pairs.csv', header=0)
        all_data = seriesFull.copy()
        all_data = bestPositions(all_data)
        X_total = sklearnFormat(all_data, fit_range, return_y = False)
        pos = model.predict(X_total)
        for i in range(len(pos)):
            self.series.loc[fit_range-1+i, 'Position'] = pos[i]
        self.series = self._calculateReturns(self.series)
