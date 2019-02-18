import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
import tensorflow as tf

class RNNRegression(BaseEstimator, RegressorMixin):
    def __init__(self, n_neurons=10, learning_rate=0.01, fit_range=50, rolling_window=10,
                       optimizer_class=tf.train.AdamOptimizer, activation=tf.nn.elu):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.fit_range = fit_range
        self.rolling_window = rolling_window
        self.steps_ahead = 1
        self.optimizer_class = optimizer_class
        self.activation = activation
        self.n_inputs = 1
        self.n_outputs = 1
        self._session = None

    def _build_graph(self):
        X = tf.placeholder(tf.float32, [None, self.fit_range, self.n_inputs])
        y = tf.placeholder(tf.float32, [None, self.fit_range, self.n_outputs])
        keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Defining RNN connections
        cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation)
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
        rnn_outputs, states = tf.nn.dynamic_rnn(cell_drop, X, dtype=tf.float32)

        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, self.fit_range, self.n_outputs])

        # Defining operations (loss, optimizer)
        loss = tf.reduce_mean(tf.square(outputs - y))
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)
        #correct = tf.nn.in_top_k(logits, y, 1)
        #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y, self._keep_prob = X, y, keep_prob
        self._outputs = outputs
        self._loss = loss
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

    def fit(self, X, max_iterations=5000, keep_prob=1.0):
        '''
        Trains RNN and saves model. Prints out cost on the test data if available
        '''
        X_train, y_train, X_test, y_test, X_total = splitData(X, fit_range = self.fit_range,
                                                              rolling_window = self.rolling_window)

        self.close_session()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()

        # needed in case of early stopping
        best_rmse = np.infty
        iterations_without_progress = 0
        if keep_prob == 1:
            max_iterations_without_progress = 60
        else:
            max_iterations_without_progress = 100
        best_params = None

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for iteration in range(max_iterations):
                sess.run(self._training_op, feed_dict={self._X: X_train, self._y: y_train, self._keep_prob: keep_prob})
                if iteration % 200 == 0:
                    rmse = np.sqrt(self._loss.eval(feed_dict={self._X: X_test, self._y: y_test, self._keep_prob: 1}))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = self._get_model_params()
                        iterations_without_progress = 0
                    else:
                        iterations_without_progress += 20
                    print("Iteration {0} - model RMSE:{1:.5f} - best RMSE:{2:.5f}".format(iteration, rmse, best_rmse))
                    if iterations_without_progress > max_iterations_without_progress:
                        print("Early stopping!")
                        break
            if best_params:
                self._restore_model_params(best_params)
        return self

    def predict(self, X):
        X_train, y_train, X_test, y_test, X_total = splitData(X, fit_range = self.fit_range,
                                                              rolling_window = self.rolling_window)
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._outputs.eval(feed_dict={self._X: X_total, self._keep_prob: 1})

    def score(self, X):
        X_train, y_train, X_test, y_test, X_total = splitData(X, fit_range = self.fit_range,
                                                              rolling_window = self.rolling_window)
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            cost = np.sqrt(self._loss.eval(feed_dict={self._X: X_test, self._y: y_test, self._keep_prob: 1}))
            return(-cost)

    def rolling_estimate(self, XInput):
        X = XInput.copy()
        rolling_mean = X['RATIO'].rolling(self.rolling_window).mean()
        rolling_std = X['RATIO'].rolling(self.rolling_window).std()
        X['RATIO_stdUnits'] = (X['RATIO']-rolling_mean)/rolling_std
        res = self.predict(X)
        for t in range(self.rolling_window+self.fit_range, len(X)):
            X.loc[X.index[0]+t+1,'RNN'] = res[t-self.fit_range-self.rolling_window][-1][0]
        return(X)

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

    def positionAccuracy(self, XInput):
        ''' Sets move (position in the strategy) and calculates its accuracy '''
        X = XInput.copy()
        try:
            for t in range(self.fit_range, len(X)):
                if X.loc[X.index[0]+t,'RNN'] >= X.loc[X.index[0]+t-1,'RNN'] :
                    X.loc[X.index[0]+t,'Position'] = 1
                else:
                    X.loc[X.index[0]+t,'Position'] = -1
            X = addNextMove(X)
        except:
            print('Need to run rolling_estimate method first')
        return(X)

def splitData(seriesInput, fit_range, rolling_window, ratio = 0.7):
    '''
    Creates X_train, y_train, X_test, y_test, X_total
    '''
    seed(10)
    series = seriesInput.copy()
    rolling_mean = series['RATIO'].rolling(rolling_window).mean()
    rolling_std = series['RATIO'].rolling(rolling_window).std()
    series['RATIO_stdUnits'] = (series['RATIO']-rolling_mean)/rolling_std
    lenSeries = len(series)-fit_range-1
    indices = sample(range(rolling_window, lenSeries), lenSeries-rolling_window)
    train_idx, test_idx = indices[:int(ratio*lenSeries)], indices[int(ratio*lenSeries):]
    ratio = np.asarray(series['RATIO_stdUnits'])
    X_total = []
    for t in range(rolling_window, len(ratio)-fit_range):
        X_total.append( ratio[t:(t+fit_range)] )
    X_train = []
    y_train = []
    for t in train_idx:
         X_train.append( ratio[t:(t+fit_range)] )
         y_train.append( ratio[(t+1):(t+fit_range+1)] )
    X_test = []
    y_test = []
    for t in test_idx:
         X_test.append( ratio[t:(t+fit_range)] )
         y_test.append( ratio[(t+1):(t+fit_range+1)] )
    return np.array(X_train).reshape(-1, fit_range, 1), np.array(y_train).reshape(-1, fit_range, 1), \
           np.array(X_test).reshape(-1, fit_range, 1),  np.array(y_test).reshape(-1, fit_range, 1), \
           np.array(X_total).reshape(-1, fit_range, 1)

def addNextMove(seriesInput):
    ''' Adds a 1 or -1 indicating if next ratio move is up or down '''
    series = seriesInput.copy()
    for t in range(len(series)-1):
        if series.loc[series.index[0]+t+1,'RATIO'] >= series.loc[series.index[0]+t,'RATIO']:
            series.loc[series.index[0]+t,'Next_move'] = 1
        else:
            series.loc[series.index[0]+t,'Next_move'] = -1
    return series
