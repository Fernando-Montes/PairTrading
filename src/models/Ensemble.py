import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from src.models.utilities import *
from src.models.RNNClassification import *
from src.models.LSTMClassification import *
from src.models.BasicStrategy import *
from sklearn.metrics import accuracy_score

class ensembleStrategy(basicStrategy):

    def apply(self, model = 'Ensemble', \
                includeRF = False, includeLR = False, includeBSC = False):
        '''
        Adds best possible entry and exit positions
         and add returns after applying strategy
        '''
        df = ensemblePositions(self.series, model, includeRF = includeRF, includeLR = includeLR,
                                includeBSC = includeBSC)
        pos = df[model]
        for i in range(len(pos)):
            self.series.loc[self.series.index[0]+19+i, 'Position'] = pos[i]
        self.series = self._calculateReturns(self.series)

def ensemblePositions(seriesInput, model, includeRF = False, includeLR = False, includeBSC = False):
    position_df = pd.DataFrame()

    if (model == 'Ensemble' and includeLR == True) or model == 'LogisticRegression':
        X = sklearnFormat(seriesInput, fit_range = 10, return_y = False)
        model_log_clf = joblib.load('../models/LogisticRegressionClassification-best')
        # RNN and LSTM are trained with 20 days previous data
        # Since the model is trained with 10 days data, only start 10 days later
        position_df['LogisticRegression'] = model_log_clf.predict(X)[10:]

    if (model == 'Ensemble' and includeRF == True) or model == 'RandomForest':
        X = sklearnFormat(seriesInput, fit_range = 10, return_y = False)
        model_rnd_clf = joblib.load('../models/RandomForestClassification-best')
        # RNN and LSTM are trained with 20 days previous data
        # Since the model is trained with 10 days data, only start 10 days later
        position_df['RandomForest'] = model_rnd_clf.predict(X)[10:]

    if model == 'Ensemble' or model == 'RNN':
        rnnCl = RNNClassification(n_neurons=20, learning_rate=0.001, fit_range=20)
        rnnCl.restore_model('../models/RNNClassification-best')
        position_df['RNN'] = rnnCl.predict(seriesInput, threshold = 0.5)

    if model == 'Ensemble' or model == 'LSTM':
        lstmCl = LSTMClassification(n_neurons=5, learning_rate=0.001, fit_range=20)
        lstmCl.restore_model('../models/LSTMClassification-best')
        position_df['LSTM'] = lstmCl.predict(seriesInput, threshold = 0.5)

    if (model == 'Ensemble' and includeBSC == True) or model == 'BasicStrategy':
        basic_str = basicStrategy(seriesInput)
        # 5-day rolling window, 0.5-std entry point, back to the mean for exit point
        position_df['BasicStrategy'] = basic_str.positions(5, 0.5, 0)[19:]

    if model == 'Ensemble' :
        for i in range(len(position_df)) :
            sum_pos =  position_df.loc[i, 'RNN'] + position_df.loc[i, 'LSTM']
            if includeRF == True:
                sum_pos = sum_pos + position_df.loc[i, 'RandomForest']
            if includeLR == True:
                sum_pos = sum_pos + position_df.loc[i, 'LogisticRegression']
            if includeBSC == True:
                sum_pos = sum_pos + position_df.loc[i, 'BasicStrategy']
            if sum_pos > 0 :
                position_df.loc[i, 'Ensemble'] = 1
            else :
                position_df.loc[i, 'Ensemble'] = -1
    return position_df

def trainModelsEnsemble(seriesInput, fit_RNN = False, fit_LSTM = False):
    '''
    Trains ensemble of classifications methods
    Returns trained methods
    '''
    X, y = sklearnFormat(seriesInput, fit_range = 10)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

    log_clf = LogisticRegression(solver="liblinear", random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42, \
                 max_depth = 3, min_samples_leaf = 15)
    scaler = StandardScaler()

    model_log_clf = pipeline.Pipeline([('scal', scaler), ('clf', log_clf)])
    model_log_clf.fit(X_train, y_train)
    joblib.dump(model_log_clf, '../models/LogisticRegressionClassification-best')

    model_rnd_clf = pipeline.Pipeline([('scal', scaler), ('clf', rnd_clf)])
    model_rnd_clf.fit(X_train, y_train)
    joblib.dump(model_rnd_clf, '../models/RandomForestClassification-best')

    train_data = bestPositions(seriesInput)
    if fit_RNN == True:
        rnnCl = RNNClassification(n_neurons=20, learning_rate=0.001, fit_range=20)
        rnnCl.fit(train_data, max_iterations=10000)
        #rnnCl.save("../models/RNNClassification-best")

    if fit_LSTM == True:
        lstmCl = LSTMClassification(n_neurons=5, learning_rate=0.001, fit_range=20)
        lstmCl.fit(train_data, max_iterations=5000)
        #lstmCl.save("../models/LSTMClassification-best")

    return None

def checkAccuracy(all_data, predictions):
    '''
    Returns accuracies for all models between ideal positions and predicted positions
    '''
    predictions['IdealPosition'] = np.array(all_data.iloc[19:,5])
    predictions = predictions.dropna()
    for i in predictions.columns.values:
        print( '{0} accuracy: {1}'.format(i, accuracy_score(predictions[i], predictions['IdealPosition'])) )
    return None
