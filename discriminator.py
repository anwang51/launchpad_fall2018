"""An LSTM network for discriminating between generated and real music.
"""

class LSTM():
    def __init__(self):
        self.input_size = 32
        self.state_size = 256
        self.num_layers = 2
        self.batch_size = 64
        self.checkpoint_dir = "./checkpoint"
        self.sess = tf.Session()
        self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        layer1 = keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid')

    def train(self):
        return
