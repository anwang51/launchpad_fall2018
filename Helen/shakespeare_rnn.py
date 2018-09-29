import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pdb

class Net:
	def __init__(self):
		self.input_size = 32
		self.output_size = 32
		self.hidden_size = 128
		self.sess = tf.Session()
		self.h_size = 100
		self.outputs = []
		self.exit_char = np.zeros(32)
		self.exit_char[26] = 1

	def _build_model(self):
		self.x = tf.placeholder(tf.float32, [None, self.input_size])
		self.h = tf.placeholder(tf.float32, [None, self.h_size])
		self.y_truth = tf.placeholder(tf.float32, [None, self.output_size])

		self.W1 = tf.Variable(np.random.rand(self.input_size + self.h_size, self.hidden_size), dtype=tf.float32)
		self.b1 = tf.Variable(np.zeros((1,self.input_size + self.h_size)), dtype=tf.float32)

		self.W2 = tf.Variable(np.random.rand(self.input_size + self.h_size, self.hidden_size), dtype=tf.float32)
		self.b2 = tf.Variable(np.zeros((1,self.input_size + self.h_size)), dtype=tf.float32)

		self.W3 = tf.Variable(np.random.rand(self.input_size + self.h_size, self.hidden_size), dtype=tf.float32)
		self.b3 = tf.Variable(np.zeros((1,self.input_size + self.h_size)), dtype=tf.float32)

		# self.y_hat = tf.placeholder(tf.float32, [None, self.output_size])

	def update(self, x_mat):
		for char in x_mat:
			print("updating with char " + str(char))
			in_vect = tf.concat(1, [char, self.h])
			next_state = tf.tanh(np.matmul(in_vect, self.W1) + self.b1)
			next_state = tf.tanh(np.matmul(next_state, self.W2) + self.b2)
			next_state = tf.tanh(np.matmul(next_state, self.W3) + self.b3)
			out = self.sess.run(next_state)
			self.outputs.append(out[:self.input_size])
			self.h = out[self.input_size:]
		y_mat = np.vstack((x_mat[1:], self.exit_char))
		losses = [tf.nn.softmax_cross_entropy_with_logits_v2(outputs[i], y_mat[i]) for i in range(len(self.outputs))]
		total_loss = tf.reduce_mean(losses)
		tf.train.AdagradOptimizer(0.1).minimize(total_loss)

	def train(self):
		while True:
			self.update(parse_text())

	def evaluate(self, x):
		self.outputs = []
		while x != self.exit_char:
			in_vect = np.hstack(char, self.h)
			next_state = tf.tanh(np.matmul(in_vect, W1) + b1)
			next_state = tf.tanh(np.matmul(next_state, W2) + b2)
			next_state = tf.tanh(np.matmul(next_state, W3) + b3)
			out = self.sess.run(next_state)
			x = out[:self.input_size]
			self.outputs.append(x)
			self.h = out[self.input_size:]
		return self.outputs

def parse_text():
	with open("shakespeare.txt", "r") as f:
		lines = f.readlines()
		x = []
		i =  0
		for line in lines:
			# print("parsing text i = " +  str(i))
			if i > 150:
				break
			for char in line.lower():
				c = np.zeros(32)
				if char == '.':
					c[26] = 1
				elif char == ' ':
					c[27] = 1
				elif char == ',':
					c[28] = 1
				elif char == '?':
					c[29] = 1
				elif char == '\'':
					c[30] = 1
				elif char == ':':
					c[31] = 1
				elif char.islower(): 
					c[ord(char)-ord('a')] = 1
				x = np.hstack((x, c))
			i += 1
	return x

net = Net()
net._build_model()
net.train()
start_char = np.zeros(32)
start_char[26] = 1
print(net.evaluate(start_char))

