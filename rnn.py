import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
import random
import math
# import pdb

class Net:

	def __init__(self):
		self.num_classes = 32
		self.output_size = 32
		self.state_size = 32
		self.char_count = 700
		self.sess = tf.Session()
		self._build_model()
		self.training_loss = []
		self.outputs = []

	def _build_model(self):
		self.x = tf.placeholder(tf.float32, [1, self.num_classes])
		self.y_truth = tf.placeholder(tf.float32, [1, self.num_classes])
		self.current_state = tf.Variable(np.zeros((1,self.state_size)), dtype=tf.float32)

		self.W_xh = tf.Variable(np.random.rand(self.num_classes, self.state_size), dtype=tf.float32)
		self.W_hh = tf.Variable(np.random.rand(self.state_size, self.state_size),dtype=tf.float32)
		self.b_h = tf.Variable(np.zeros((1,self.state_size)), dtype=tf.float32)

		self.W_hy = tf.Variable(np.random.rand(self.state_size, self.num_classes),dtype=tf.float32)
		self.b_y = tf.Variable(np.zeros((1,self.num_classes)), dtype=tf.float32)

		self.next_state = tf.tanh(tf.matmul(self.x, self.W_xh) + tf.matmul(self.current_state, self.W_hh) + self.b_h)
		self.output = tf.nn.softmax(tf.matmul(self.next_state, self.W_hy) + self.b_y)
		# print(self.output)
		# print(self.y_truth)

		self.losses = tf.losses.softmax_cross_entropy(onehot_labels=self.y_truth,logits=self.output)
		self.total_loss = tf.reduce_mean(self.losses)
		self.optimizer = tf.train.AdagradOptimizer(0.3).minimize(self.total_loss)
		# self.loss = tf.losses.(self.output, self.y_truth)
		# self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

		# DO NOT TOUCH BELOW
		self.sess.run(tf.global_variables_initializer())

	def update(self, x_mat, y_mat):
		return self.sess.run([self.losses, self.optimizer, self.output], {self.x: x_mat, self.y_truth: y_mat})

	def one_hotter(self, letter):
		self.alphabets = {'a' : 0, 'b': 1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10, 'l':11,
		'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24,
		'z': 25, '.': 26, '!':27, '?':28, ' ':29, ',':30, '%':31}
		if letter in self.alphabets:
			result = np.zeros(32)
			result[self.alphabets[letter]] = 1
			return result
			# return tf.one_hot(alphabets[letter], 32, on_value=1, off_value=0).eval(session=self.sess)

	def train(self):
		text = open('shakespeare.txt','r')
		shakespeare = []
		for line in text:
			shakespeare.append(line.replace('\n','').strip())
		stopchar = False
		index = 0 # what line
		for i in range(396*1000):
			letter_vectors = []
			while not stopchar:
				letters = list(shakespeare[index])
				for letter in letters:
					a = self.one_hotter(letter)
					if a is not None:
						letter_vectors.append(a)
					if letter in ['.','?','!']:
						stopchar = True
						letter_vectors.append(self.one_hotter("%"))
				index += 1
			if (i+1) % 396 == 0:
				index = 0
			letter_vectors = np.array(letter_vectors)
			x_mat = np.zeros([self.char_count,self.num_classes])
			x_mat[:letter_vectors.shape[0]-1,:letter_vectors.shape[1]] = letter_vectors[:-1]
			y_mat = np.zeros([self.char_count,self.num_classes])
			y_mat[:letter_vectors.shape[0]-1,:letter_vectors.shape[1]] = letter_vectors[1:]
			for i in range(self.char_count):
				loss, _, output = self.update([x_mat[i]], [y_mat[i]])
			stopchar = False
			# if counter > 0 and counter % 1000 == 0:
			# 	self.training_loss.append(loss)
			# 	self.outputs.append(output)

	def evaluate(self, x, count):
		letter = self.sess.run(self.output, {self.x: x})
		character = list(self.alphabets.keys())[list(self.alphabets.values()).index(letter.argmax())]
		print(character, end='')
		if character != "%" and count < 500:
			return self.evaluate(self.sess.run(self.output, {self.x: x}), count + 1)
		else:
			print()

# n = 10000
# x_mat = [2*math.pi*(float(i) / n) for i in range(n)]
sin_pred = Net()
