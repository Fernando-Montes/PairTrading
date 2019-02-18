from src.models.BasicStrategy import *
from src.models.utilities import *

class bestPositionsStrategy(basicStrategy):

    def apply(self):
        '''
        Adds best possible entry and exit positions
         and add returns after applying strategy
        '''
        self.series = bestPositions(self.series)
        self.series.rename(columns={'IdealPosition': 'Position'}, inplace=True)
        self.series = self._calculateReturns(self.series)
