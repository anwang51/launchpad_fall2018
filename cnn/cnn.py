import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pdb

class Net:

	def __init__(self):
		self.input_size = 1
		self.output_size = 1
		self.hidden_size = 128
		self.sess = tf.Session()
		self._build_model()
		self.training_loss = []
		self.outputs = []

	def _build_model(self):
		self.x = tf.placeholder(tf.float32, [None, self.input_size])
		self.y_truth = tf.placeholder(tf.float32, [None, self.output_size])
		layer1 = tf.layers.dense(self.x, self.hidden_size, activation=tf.nn.relu)
		self.y_hat = tf.layers.dense(layer1, self.output_size, activation=tf.tanh)
		self.loss = tf.losses.mean_squared_error(self.y_hat, self.y_truth)
		self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
		
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
				x = random.uniform(0, 2*math.pi)
				y = np.sin(x)
				x = np.array([x])
				y = np.array([y])
				x_mat.append(x)
				y_mat.append(y)
			x_mat = np.array(x_mat)
			y_mat = np.array(y_mat)
			counter += 1
			loss, _, output = self.update(x_mat, y_mat)
			if counter > 0 and counter % 1000 == 0:
				self.training_loss.append(loss)
				self.outputs.append(output)

	def evaluate(self, x):
		return self.sess.run(self.y_hat, {self.x: x})

n = 10000
x_mat = [2*math.pi*(float(i) / n) for i in range(n)]
sin_pred = Net()

