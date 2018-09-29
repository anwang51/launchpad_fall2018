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
		self.state_size = 100 #hidden size
		self.truncated_backprop_length = 15 # how many characters to look fowards
		self.char_count = 700
		self.sess = tf.Session()
		self._build_model()
		self.training_loss = []
		self.outputs = []

	def _build_model(self):
		self.x = tf.placeholder(tf.float32, [None, self.num_classes])
		self.y_truth = tf.placeholder(tf.float32, [None, self.num_classes])
		self.current_state = tf.Variable(np.zeros((1,self.state_size)), dtype=tf.float32)

		self.W_xh = tf.Variable(np.random.rand(self.num_classes, self.state_size), dtype=tf.float32)
		self.W_hh = tf.Variable(np.random.rand(self.state_size, self.state_size),dtype=tf.float32)
		self.b_h = tf.Variable(np.zeros((1,self.state_size)), dtype=tf.float32)

		self.W_hy = tf.Variable(tf.random_normal(self.state_size, self.num_classes,stddev=0.01),dtype=tf.float32)
		self.b_y = tf.Variable(np.zeros((1,self.num_classes)), dtype=tf.float32)

		self.next_state = tf.tanh(tf.matmul(self.x, self.W_xh) + tf.matmul(self.current_state, self.W_hh) + self.b_h)
		self.output = tf.matmul(self.next_state, self.W_hy) + self.b_y

		# print(self.output.shape)
		# print(self.y_truth.shape)

		self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=self.y_truth)
		self.total_loss = tf.reduce_mean(self.losses)
		# self.optimizer = tf.train.AdagradOptimizer(0.3).minimize(self.total_loss)
		# self.loss = tf.losses.(self.output, self.y_truth)
		self.optimizer = tf.train.AdamOptimizer().minimize(self.total_loss)

		# DO NOT TOUCH BELOW
		self.sess.run(tf.global_variables_initializer())

	def update(self, x_mat, y_mat):
		return self.sess.run([self.total_loss, self.optimizer, self.output,self.next_state], {self.x: x_mat, self.y_truth: y_mat})

	def one_hotter(self, letter):
		self.alphabets = {'a' : 0, 'b': 1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10, 'l':11,
		'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24,
		'z': 25, '.': 26, '!':27, '?':28, ' ':29, ',':30, '%':31}
		if letter in self.alphabets:
			result = np.zeros(32) # leave the first empty for nonsense/padding
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
			x_vec = letter_vectors[:-1]
			y_vec = letter_vectors[1:]
			# x_mat = np.zeros([self.char_count,self.num_classes])
			# x_mat[:letter_vectors.shape[0]-1,:letter_vectors.shape[1]] = letter_vectors[:-1]
			# y_mat = np.zeros([self.char_count,self.num_classes])
			# y_mat[:letter_vectors.shape[0]-1,:letter_vectors.shape[1]] = letter_vectors[1:]
			for e in range(letter_vectors.shape[0]//self.truncated_backprop_length):
				start_idx = e * self.truncated_backprop_length
				end_idx = start_idx + self.truncated_backprop_length
				# x_mat = np.zeros([self.truncated_backprop_length,self.num_classes])
				# x_mat[:x_vec.shape[0],:x_vec.shape[1]] = x_vec[start_idx:end_idx]
				x_mat = x_vec[start_idx:end_idx]
				x_mat = np.pad(x_mat,((0,self.truncated_backprop_length-len(x_mat)),(0,0)),'constant')
				y_mat = y_vec[start_idx:end_idx]
				y_mat = np.pad(y_mat,((0,self.truncated_backprop_length-len(y_mat)),(0,0)),'constant')
				# y_mat = np.zeros([self.truncated_backprop_length,self.num_classes])
				# y_mat[:y_vec.shape[0],:y_vec.shape[1]] = y_vec[start_idx:end_idx]
				# print(start_idx,end_idx)
				loss, _, output, self.current_state = self.update(x_mat,y_mat)
			stopchar = False
			# print(i)
			if i > 0 and i % 396 == 0:
				# print(output,y_mat)
				print("i: ",i, "Loss: ",loss)
				self.training_loss.append(loss)
				self.outputs.append(output)
				# print(self.sample(self.current_state[24], 0, 200))

	def evaluate(self, x, count):
		letter = self.sess.run(self.output, {self.x: x})
		character = list(self.alphabets.keys())[list(self.alphabets.values()).index(letter[self.truncated_backprop_length-1].argmax())]
		print(character, end='')
		if character != "%" and count < 500:
			# print(np.vstack([x[1:],self.one_hotter(character)]).shape)
			return self.evaluate(np.vstack([x[1:],self.one_hotter(character)]), count + 1)
		else:
			print()

	# def sample(self, h, seed_ix, n):
	# 	x = np.zeros((self.num_classes, 1))
	# 	x[seed_ix] = 1
	# 	ixes = []
	# 	print(np.dot(x, self.W_xh).shape, np.dot(h, self.W_hh).shape,self.b_h.shape)
	# 	for t in range(n):
	# 		h = np.tanh(np.dot(x, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
	# 		y = np.dot(h, self.W_hy) + self.b_y
	# 		p = np.exp(y) / np.sum(np.exp(y))
	# 		ix = np.random.choice(range(self.num_classes), p=p.ravel())
	# 		x = np.zeros((self.num_classes, 1))
	# 		x[ix] = 1
	# 		ixes.append(ix)
	# 	return ixes

# n = 10000
# x_mat = [2*math.pi*(float(i) / n) for i in range(n)]
sin_pred = Net()
a = []
a.append(sin_pred.one_hotter('h'))
a.append(sin_pred.one_hotter('e'))
a.append(sin_pred.one_hotter('l'))
a.append(sin_pred.one_hotter('l'))
a.append(sin_pred.one_hotter('o'))
a.append(sin_pred.one_hotter('h'))
a.append(sin_pred.one_hotter('e'))
a.append(sin_pred.one_hotter('l'))
a.append(sin_pred.one_hotter('l'))
a.append(sin_pred.one_hotter('o'))
a.append(sin_pred.one_hotter('h'))
a.append(sin_pred.one_hotter('e'))
a.append(sin_pred.one_hotter('l'))
a.append(sin_pred.one_hotter('l'))
a.append(sin_pred.one_hotter('o'))
a = np.array(a)
