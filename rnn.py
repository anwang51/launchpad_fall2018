import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pdb



class Net:

	def __init__(self):
		self.input_length = 128
		self.one_hot_size = 0 # set later in encode_data()
		self.text_len = 5000 # also set later
		self.encode_data()
		self.h_size = 256
		self.hidden_size = 256
		self.sess = tf.Session()
		self._init_weights()
		self.build_model()
		self.training_loss = []
		self.outputs = []
		self.num_letters = 2000

	def _init_weights(self):
		# self.x = tf.placeholder(tf.float32, [self.text_len, self.one_hot_size])
		self.xs = [tf.placeholder(tf.float32, [1, self.one_hot_size]) for _ in range(self.text_len)]
		self.y_truth = tf.placeholder(tf.float32, [self.text_len, self.one_hot_size])

		self.W1 = tf.Variable(np.random.rand(self.one_hot_size + self.h_size, self.hidden_size), dtype=tf.float32)
		self.b1 = tf.Variable(np.zeros((1, self.hidden_size)), dtype=tf.float32)

		self.W2 = tf.Variable(np.random.rand(self.hidden_size, self.hidden_size), dtype=tf.float32)
		self.b2 = tf.Variable(np.zeros((1, self.hidden_size)), dtype=tf.float32)

		self.W3 = tf.Variable(np.random.rand(self.hidden_size, self.one_hot_size + self.h_size), dtype=tf.float32)
		self.b3 = tf.Variable(np.zeros((1, self.one_hot_size + self.h_size)), dtype=tf.float32)

		# DO NOT TOUCH BELOW
		self.sess.run(tf.global_variables_initializer())

	def encode_data(self):
		contents = open("shakespeare.txt").read().lower()[:5000]
		chars = list(set(contents))
		chars.sort()
		self.one_hot_size = len(chars) + 1
		self.text_len = len(contents)
		char_dict = {}
		for i, c in enumerate(chars):
			char_dict[c] = i
		contents_ind = np.array([char_dict[c] for c in contents])
		self.x_mat = np.zeros((len(contents), len(chars) + 1))
		self.x_mat[np.arange(len(contents)), contents_ind] = 1
		self.y_mat = np.append(x_mat[1:], np.array([0 for _ in range(len(chars))] + [1]))

	def build_model(self):

		h = tf.convert_to_tensor(np.array([np.zeros(self.h_size)]), dtype=tf.float32)
		prediction = []
		# for x_char in tf.unstack(self.x, axis=0):
		for x_char in self.xs:

			# pdb.set_trace()
			x_final = tf.concat([x_char, h], 1)
			layer1 = tf.nn.relu(tf.matmul(x_final, self.W1) + self.b1)
			layer2 = tf.nn.relu(tf.matmul(layer1, self.W2) + self.b2)
			layer3 = tf.nn.relu(tf.matmul(layer2, self.W3) + self.b3)
			combined_out = self.sess.run(layer3)
			if len(combined_out) != (self.one_hot_size + self.h_size):
				raise Error("sizes no match you dumb")
			self.y_hat = [tf.nn.softmax(i) for i in combined_out[0:self.one_hot_size]]
			prediction.append(np.array(y_hat))
			h = combined_out[self.one_hot_size:]
		labels = tf.unstack(self.y_truth, axis=0)
		losses = [tf.nn.softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(prediction, labels)]
		self.loss = tf.reduce_mean(losses)
		self.optimizer = tf.train.AdamOptimizer().minimize(loss)

	def update(self, x_mat, y_mat):
		self.sess.run([self.loss, self.optimizer], {self.x: x_mat, self.y_truth: y_mat})

	def train(self):
		counter = 0 
		while True:
			counter += 1
			loss, _ = self.update(x_mat, y_mat)
			if counter > 0 and counter % 1000 == 0:
				self.training_loss.append(loss)

	def evaluate(self, x):
		return self.sess.run(self.y_hat, {self.x: x})

n = 10000
x_mat = [2*math.pi*(float(i) / n) for i in range(n)]
sin_pred = Net()

