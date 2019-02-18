from src.models.RNNClassification import *

class LSTMClassification(RNNClassification):
    def __init__(self, n_neurons=20, n_layers=1, learning_rate=0.01, fit_range=20,
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

        # Defining LSTM connections
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_neurons, use_peepholes=True)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
        h_state = states[1] # Only passes h-state to final cell

        logits = tf.layers.dense(h_state, self.n_outputs)
        Y_proba = tf.nn.softmax(logits)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)

        # Defining operations (loss, optimizer)
        loss = tf.reduce_mean(xentropy)
        optimizer = self.optimizer_class(learning_rate=self.learning_rate, beta1=0.93)
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
