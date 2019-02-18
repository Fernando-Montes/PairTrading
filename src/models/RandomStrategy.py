from src.models.BasicStrategy import *

class randomStrategy(basicStrategy):

    def apply(self):
        '''
        Adds entry and exit positions at random times
         and add returns after applying strategy
        '''
        self.series['Position'] = np.random.choice([1,0,-1], size = len(self.series), p=[0.1, 0.8, 0.1])
        self.series = self._calculateReturns(self.series)

    def histogram(self, num_tries = 100):
        histTrain = []
        histTest = []
        for i in range(num_tries):
            self.apply()
            apr_train, apr_test = self.apr()
            histTrain.append(apr_train)
            histTest.append(apr_test)
        plotHisto(histTrain, 'AR train data [%]', bins = 20)
