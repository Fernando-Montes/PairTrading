import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from random import sample, seed
from src.models.utilities import *

class RNNClassification(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=20, learning_rate=0.001, fit_range=20,
                       optimizer_class=tf.train.AdamOptimizer):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.fit_range = fit_range
        self.optimizer_class = optimizer_class
        self.n_inputs = 4
        self.n_outputs = 2
        self._session = None

    def _build_graph(self):
        X = tf.placeholder(tf.float32, [None, self.fit_range, self.n_inputs])
        y = tf.placeholder(tf.int32, [None])
        keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Defining RNN connections
        cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons)
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
        outputs, states = tf.nn.dynamic_rnn(cell_drop, X, dtype=tf.float32)

        logits = tf.layers.dense(states, self.n_outputs)
        Y_proba = tf.nn.softmax(logits)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)

        # Defining operations (loss, optimizer)
        loss = tf.reduce_mean(xentropy)
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y, self._keep_prob = X, y, keep_prob
        self._outputs = outputs
        self._correct = correct
        self._Y_proba = Y_proba
        self._loss = loss
        self._accuracy = accuracy
        self._training_op = training_op
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, max_iterations=10000, keep_prob=1.0):
        '''
        Trains RNN
        '''
        X_train, y_train, X_test, y_test, X_total, y_total = createTrainValidation(X, fit_range = self.fit_range)

        self.close_session()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()

        # needed in case of early stopping
        best_acc_test = 0
        iterations_without_progress = 0
        if keep_prob == 1:
            max_iterations_without_progress = 20000
        else:
            max_iterations_without_progress = 20000
        best_params = None

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for iteration in range(max_iterations):
                sess.run(self._training_op, feed_dict={self._X: X_train, self._y: y_train, self._keep_prob: keep_prob})
                acc_train = self._accuracy.eval(feed_dict={self._X: X_train, self._y: y_train})
                acc_test = self._accuracy.eval(feed_dict={self._X: X_test, self._y: y_test})
                if acc_test > best_acc_test:
                    best_acc_test = acc_test
                    best_params = self._get_model_params()
                    iterations_without_progress = 0
                    print("Iter {0} - model acc_train:{1:.4f} - model acc_validation:{2:.4f} - best acc_validation:{3:.4f}".format( \
                            iteration, acc_train, acc_test, best_acc_test))
                else:
                    iterations_without_progress += 1
                if iteration % 500 == 0:
                    print("Iter {0} - model acc_train:{1:.4f} - model acc_validation:{2:.4f} - best acc_validation:{3:.4f}".format( \
                            iteration, acc_train, acc_test, best_acc_test))
                    if iterations_without_progress > max_iterations_without_progress:
                        print("Early stopping!")
                        break
            if best_params:
                self._restore_model_params(best_params)
        return self

    def predict_proba(self, X):
        ''' Predicts softmax probabilities '''
        X_total = formatData(X, fit_range = self.fit_range)
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X_total, self._keep_prob: 1})

    def predict(self, X, threshold = 0.5):
        ''' Sets the predicted position if the max probability is above threshold '''
        prob = self.predict_proba(X)
        positions = []
        for i in range(len(prob)):
            if prob[i,1] >= prob[i,0] and prob[i,1] >= threshold :
                positions.append(1)
            elif prob[i,0] > prob[i,1] and prob[i,0] > threshold :
                positions.append(-1)
            else :
                positions.append(0)
        return np.array(positions)

    def score(self, X):
        ''' Used by gridSearch '''
        X_train, y_train, X_test, y_test, X_total, y_total = createTrainValidation(X, fit_range = self.fit_range)
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            acc_test = self._accuracy.eval(feed_dict={self._X: X_total, self._y: y_total})
            return(acc_test)

    def check(self, X):
        ''' Used for debugging '''
        X_train, y_train, X_test, y_test, X_total, y_total = createTrainValidation(X, fit_range = self.fit_range)
        with self._session.as_default() as sess:
            acc_train = self._accuracy.eval(feed_dict={self._X: X_train, self._y: y_train})
            acc_test = self._accuracy.eval(feed_dict={self._X: X_test, self._y: y_test})
            acc_total = self._accuracy.eval(feed_dict={self._X: X_total, self._y: y_total})
            print("Best model: acc_train:{0:.4f} - acc_validation:{1:.4f} - acc_total:{2:.4f}".format( \
                    acc_train, acc_test, acc_total))
            correct = self._correct.eval(feed_dict={self._X: X_total, self._y: y_total})
            print(correct)

    def save(self, path="../models/RNNmodel"):
        self._saver.save(self._session, path)

    def restore_model(self, path="../models/RNNmodel"):
        self.close_session()
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            try:
                self._saver.restore(sess, path)
            except:
                print('Does the file exist? did you set the hyper-parameters the same as in the save?')
        return self

def createTrainValidation(seriesInput, fit_range, ratio = 0.7):
    '''
    Creates Train and validation data in RNN input format
    '''
    seed(10)
    series = seriesInput.copy()
    position = np.array(series['IdealPosition'])
    series = series[['RATIO', 'DIFF', 'ABC', 'XYZ']]
    for t in range(len(series)):
        if position[t] == -1 : # Need to do this since the labels need to be >=0
            position[t] = 0
    len_data = len(series)-fit_range-1
    indices = sample(range(len_data), len_data)
    train_idx, test_idx = indices[:int(ratio*len_data)], indices[int(ratio*len_data):]
    X_train = []
    y_train = []
    for t in train_idx:
        X_train.append( np.array(series[t:(t+fit_range)]) )
        y_train.append( position[t+fit_range-1] )
    X_test = []
    y_test = []
    for t in test_idx:
        X_test.append( np.array(series[t:(t+fit_range)]) )
        y_test.append( position[t+fit_range-1] )
    X_total = []
    y_total = []
    for t in range(len(series)-fit_range):
         X_total.append( np.array(series[t:(t+fit_range)]) )
         y_total.append( position[t+fit_range-1] )

    return np.array(X_train), np.array(y_train).reshape(-1), \
           np.array(X_test), np.array(y_test).reshape(-1), \
           np.array(X_total), np.array(y_total).reshape(-1)
