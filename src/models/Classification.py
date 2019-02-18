import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import pipeline
from src.models.utilities import *

def trainClassification(seriesInput, fit_range = 10):
    '''
    Trains ensemble of classifications methods
    Returns trained method
    '''
    X, y = sklearnFormat(seriesInput, fit_range)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

    log_clf = LogisticRegression(solver="liblinear", random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42, \
                max_depth = 3, min_samples_leaf = 15)
    knn_clf = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform')

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('kn', knn_clf)],
        voting='hard')
    scaler = preprocessing.StandardScaler()
    voting_sc_clf = pipeline.Pipeline([('scal', scaler), ('vo', voting_clf)])
    voting_sc_clf.__class__.__name__ = 'VotingClassifier'

    for clf in (log_clf, rnd_clf, knn_clf, voting_sc_clf):
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_valid = clf.predict(X_valid)
        print(' Model {0} - acc. train set: {1:.4f} - acc. validation set: {2:.4f}'.format( \
                    clf.__class__.__name__, accuracy_score(y_train, y_pred_train), \
                    accuracy_score(y_valid, y_pred_valid)) )
    return voting_sc_clf
