"""An LSTM network for discriminating between generated and real music.
(maybe we should use DRNN?)
"""

class LSTM():
    def __init__(self):
        self.input_size = # some variable number
        self.state_size = 256
        self.num_layers = 2
        self.batch_size = 64
        self.sess = tf.Session()
        self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        self.layer1 = keras.layers.LSTM(self.input_size, activation='tanh', recurrent_activation='hard_sigmoid')
        # maybe not with keras?

    def train(self):
        return
