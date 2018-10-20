import tensorflow as tf
import numpy as np
import os
import random

# does not do end-of-sentence

class LSTM:

    def __init__(self):
        self.num_notes = 128 #in size
        self.output_size = 128 #out size
        self.state_size = 256 #hidden size
        self.num_layers = 2
        self.batch_size = 64 #'sentences' to look at
        self.steps = 100 #chars in a sentence ? might need to be higher, variable padding
        self.checkpoint_dir = "./checkpoint"
        self.sess = tf.Session()
        self._build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_model(self):
        self.x = tf.placeholder(tf.float32,[None, None, self.num_notes, 1])
        self.y_truth = tf.placeholder(tf.float32, [None, None, self.num_notes, 1])

        conv1 = tf.layers.conv2d(
            inputs=self.x,
            filters=32,
            kernel_size=[3,3],
            padding="same",
            activation=tf.nn.relu
            )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=[2, 2]
            )
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu
            )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2,2],
            strides=[2,2]
            )
        pool2_flat = tf.layers.Flatten()(pool2)
        pool2_flat = tf.expand_dims(pool2_flat, 2)

        self.init_state = tf.placeholder(tf.float32, [None, self.num_layers * 2 * self.state_size])
        self.lstm_cells = [tf.nn.rnn_cell.LSTMCell(self.state_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
        self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells,state_is_tuple=False)
        # Iteratively compute output of recurrent network
        outputs, self.new_state = tf.nn.dynamic_rnn(self.lstm, pool2_flat, initial_state=self.init_state, dtype=tf.float32)
        self.W_hy = tf.Variable(tf.random_normal((self.state_size, self.num_notes),stddev=0.01),dtype=tf.float32)
        self.b_y = tf.Variable(tf.random_normal((self.output_size,), stddev=0.01), dtype=tf.float32)
        net_output = tf.matmul(tf.reshape(outputs, [-1, self.state_size]), self.W_hy) + self.b_y

        lstm_output = tf.reshape(tf.nn.softmax(net_output),(tf.shape(outputs)[0], tf.shape(outputs)[1], self.output_size))
        reshaped = tf.reshape(lstm_output, [-1, 32, -1, 1])
        deconv1 = tf.nn.conv2d_transpose(reshaped, tf.placeholder(tf.float32, shape=[3, 3, 1, 1]), tf.placeholder(tf.int32, shape=(4,)), [1, 2, 2, 1], padding="SAME")
        self.final_outputs = tf.nn.conv2d_transpose(deconv1, tf.placeholder(tf.float32, shape=[3, 3, 1, 1]), tf.placeholder(tf.int32, shape=(4,)), [1, 2, 2, 1], padding="SAME")
        self.lstm_last_state = np.zeros((self.num_layers * 2 * self.state_size))

        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=net_output,labels=tf.reshape(self.y_truth, [-1, self.output_size]))
        self.total_loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.003),0.9).minimize(self.total_loss)


model = LSTM()