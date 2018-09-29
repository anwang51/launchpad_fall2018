import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pdb

class Net:

	def __init__(self):
		self.input_size = 100
		self.output_size = 1
		self.hidden_size = 128
		self.sess = tf.Session()
		self.alphabet_size = 75
		self.shakespeare_data = open("processed_shakespeare.txt").read()
		self.TEXT_SIZE = len(self.shakespeare_data)
		self.AVG_TEXT_SIZE = self.input_size
		self._build_model()
		self.training_loss = []
		self.outputs = []

	def _build_model(self):
		self.x = tf.placeholder(tf.float32, [None, self.input_size, self.alphabet_size])
		self.y_truth = tf.placeholder(tf.float32, [None, self.input_size, self.alphabet_size])
		lstm_cell_1 = tf.contrib.rnn.LSTMCell(self.alphabet_size)
		lstm_cell_2 = tf.contrib.rnn.LSTMCell(self.alphabet_size)
		multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2] , state_is_tuple=True)
		self.y_hat, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, self.x, dtype=tf.float32)
		# self.y_hat = tf.layers.dense(outputs, self.output_size, activation=tf.tanh)
		self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_truth, logits=self.y_hat)
		self.optimizer = tf.train.RMSPropOptimizer().minimize(self.loss)
		# DO NOT TOUCH BELOW
		self.sess.run(tf.global_variables_initializer())

	def update(self, x_mat, y_mat):
		return self.sess.run([self.loss, self.optimizer, self.y_hat], {self.x: x_mat, self.y_truth: y_mat})

	def train(self):
		counter = 0 
		while True:
			x_mat = []
			y_mat = []
			for _ in range(32):
				start = random.randint(0, self.TEXT_SIZE-self.AVG_TEXT_SIZE)
				hot_x = one_hotter(self.shakespeare_data[start : start+self.AVG_TEXT_SIZE])
				x_mat.append(hot_x)
			x_mat = np.array(x_mat)
			y_mat = np.array(x_mat)
			counter += 1
			loss, _, output = self.update(x_mat, y_mat)
			if counter > 0 and counter % 50 == 0:
				loss = np.mean([np.mean(l) for l in loss])
				print("Loss: %d", loss)
				self.training_loss.append(loss)
				self.outputs.append(output)

	def evaluate(self, x):
		return self.sess.run(self.y_hat, {self.x: x})

alphabet_encoding = {
	32:0, 33:1, 46:2, 44:3, 39:4, 63:5, 59:6, 10:7, 58:8, 45:9, 38:10, 91:11, 93:12
}

def one_hotter(s):
	temp = []
	for c in s:
		z = np.zeros(75)
		ind = alphabet_encoding[ord(c)]
		z[ind] = 1
		temp.append(np.array(z))
	temp = np.array(temp)
	return temp

upper_case_offset = 65
lower_case_offset = 97
upper_case = 13
lower_case = 39
for i in range(26):
	alphabet_encoding[upper_case_offset + i] = upper_case + i
	alphabet_encoding[lower_case_offset + i] = lower_case + i
numbers_offset = 48
numbers = 65
for i in range(10):
	alphabet_encoding[numbers_offset + i] = numbers + i
lstm = Net()
