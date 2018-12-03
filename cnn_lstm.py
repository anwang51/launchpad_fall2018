import tensorflow as tf
import numpy as np
import os
import random
import pdb

# does not do end-of-sentence

class LSTM:

	def __init__(self):
		self.num_notes = 128 #in size
		self.output_size = 128 #out size
		self.state_size = 256 
		self.hidden_size = 256 #hidden size
		self.num_layers = 2
		self.stride_length = 2
		self.batch_size = 64 #'sentences' to look at
		self.steps = 100 #chars in a sentence ? might need to be higher, variable padding
		self.checkpoint_dir = "./checkpoint"
		self.sess = tf.Session()
		self._build_model()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def _build_model(self):
		self.x = tf.placeholder(tf.float32,[self.batch_size, None, self.num_notes, 1])
		self.y_truth = self.x

		conv1 = tf.layers.conv2d(
			inputs=self.x,
			filters=128,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.relu
			)
		pool1 = tf.layers.max_pooling2d(
			inputs=conv1,
			pool_size=[2, 2],
			strides=[self.stride_length, self.stride_length]
			)
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=128,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.relu
			)
		pool2 = tf.layers.max_pooling2d(
			inputs=conv2,
			pool_size=[2, 2],
			strides=[self.stride_length, self.stride_length]
			)
		conv3 = tf.layers.conv2d(
			inputs=pool2,
			filters=128,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.relu
			)
		pool3 = tf.layers.max_pooling2d(
			inputs=conv3,
			pool_size=[2,2],
			strides=[self.stride_length, self.stride_length]
			)

		# pdb.set_trace()
		# pool3_flat = tf.layers.Flatten()(pool3)
		# pool3_flat = tf.expand_dims(pool3, 3)

		self.init_state = tf.placeholder(tf.float32, [self.batch_size, self.num_layers * 2 * self.state_size])
		self.lstm_cells = [tf.nn.rnn_cell.LSTMCell(self.state_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
		self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells,state_is_tuple=False)
		pdb.set_trace()
		# Iteratively compute output of recurrent network
		# test = tf.placeholder(tf.float32, [self.batch_size, self.num_notes, 256])
		# outputs, self.new_state = tf.nn.dynamic_rnn(self.lstm, test, initial_state=self.init_state, dtype=tf.float32)
		outputs, self.new_state = tf.nn.dynamic_rnn(self.lstm, pool3, initial_state=self.init_state, dtype=tf.float32)
		self.W_hy = tf.Variable(tf.random_normal((self.state_size, self.num_notes),stddev=0.01),dtype=tf.float32)
		self.b_y = tf.Variable(tf.random_normal((self.output_size,), stddev=0.01), dtype=tf.float32)
		# net_output = tf.matmul(tf.reshape(outputs, [-1, self.state_size]), self.W_hy) + self.b_y
		# net_output = tf.matmul(tf.reshape(outputs, [self.batch_size, self.state_size]), self.W_hy) + self.b_y
		net_output = tf.matmul(tf.reshape(outputs, [self.batch_size, -1, self.num_notes]), self.W_hy) + self.b_y

		lstm_output = tf.reshape(tf.nn.softmax(net_output),(tf.shape(outputs)[0], tf.shape(outputs)[1], self.output_size))
		reshaped = tf.reshape(lstm_output, [self.batch_size, int(self.num_notes/(self.stride_length * self.stride_length)), -1, 1])

		deconv1 = tf.layers.conv2d_transpose(reshaped,
			filters=128,
			kernel_size=[5,5],
			padding="same",
			strides=[self.stride_length, self.stride_length],
			activation=tf.nn.relu
			)
		deconv2 = tf.layers.conv2d_transpose(deconv1,
			filters=128,
			kernel_size=[3,3],
			padding="same",
			strides=[self.stride_length, self.stride_length],
			activation=tf.nn.relu
			)
		self.final_outputs = tf.layers.conv2d_transpose(deconv2,
			filters=1,
			kernel_size=[3,3],
			padding="same",
			strides=[self.stride_length, self.stride_length],
			activation=tf.nn.relu
			)
		self.lstm_last_state = np.zeros((self.num_layers * 2 * self.state_size))

		self.losses = tf.losses.mean_squared_error(self.final_outputs, self.y_truth) #tf.reshape(self.y_truth, [-1, self.output_size]))
		self.total_loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.005),0.995)
		self.variables = tf.trainable_variables()
		self.gradients = self.optimizer.compute_gradients(self.total_loss, self.variables)
		self.optimizer.apply_gradients(self.gradients)

	def train(self, x):
		num_timesteps = len(x)
		x = np.reshape(x, (num_timesteps, self.num_notes, 1))
		y = np.append(x[1:], np.zeros((1, self.num_notes, 1)), axis=0)

		counter = 0
		while True:
			x_mat, y_mat = [], []
			for _ in range(self.batch_size):
				x_mat.append(x)
				y_mat.append(y)
			x_mat = np.array(x_mat)
			y_mat = np.array(y_mat)
			init_state = np.random.rand(self.batch_size, self.num_layers * 2 * self.state_size)
			loss, losses, variables = self.sess.run([self.total_loss, self.losses, self.variables], {self.x: x_mat, self.init_state: init_state})
			counter +=1
			if counter % 1 == 0:
				print("Loss: %f" % (loss))

	def evaluate(self, x, length):
		# IN PROGRESS
		x = np.array([np.reshape(x, (num_timesteps, self.num_notes, 1))])
		# assume output is same size as input
		init_output, state = self.sess.run([self.final_outputs, self.lstm_last_state], {self.x: x,
			self.init_state: np.random.rand(num_timesteps, self.num_layers * 2 * self.state_size)})
		y = init_output
		last_note = np.array([init_output[0][-1]])
		for _ in range(length):
			last_note, state = self.sess.run([self.final_outputs, self.lstm_last_state], {self.x: last_note,
			self.init_state: state})
			y = np.append(y, last_note[0], axis=0)
		return y

def rand_track(num_beats):
	return np.random.rand(num_beats, 128)

if __name__ == "__main__":
	model = LSTM()
	model.train(rand_track(128))
