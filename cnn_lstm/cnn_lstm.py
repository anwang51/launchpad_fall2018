import tensorflow as tf
import numpy as np
import os
import random
import pdb
import data_io
import itertools
import time

# does not do end-of-sentence
ROOT_DIR = "lpd_5_cleansed"
SONG_LENGTH = 512
ERRORS = []
RECORD_INTERVAL = 300 # 900 seconds

class LSTM:

	def __init__(self):
		self.sess = tf.Session()
		self.num_notes = 128 #in size
		self.state_size = 256
		self.hidden_size = 256 #hidden size
		self.num_layers = 2
		self.stride_length = 2
		self.num_filters = 16
		self.conv_out_size = int(self.num_notes / (np.power(self.stride_length, 3)))
		self.conv_concat = self.conv_out_size * self.num_filters
		self.lstm_hidden_size = self.conv_out_size * self.num_filters
		self.output_size = self.lstm_hidden_size #out size
		self.batch_size = 64 #'sentences' to look at
		self.steps = 100 #chars in a sentence ? might need to be higher, variable padding
		self.checkpoint_dir = "./checkpoint"
		self._build_model()
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())

	def _build_model(self):
		# self.x = tf.placeholder(tf.float32,[self.batch_size, None , self.num_notes, 1])
		self.x = tf.placeholder(tf.float32,[self.batch_size, None , self.num_notes, 1])
		self.y_truth = self.x

		conv1 = tf.layers.conv2d(
			inputs=self.x,
			filters=self.num_filters,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.leaky_relu
			)
		pool1 = tf.layers.max_pooling2d(
			inputs=conv1,
			pool_size=[2, 2],
			strides=[self.stride_length, self.stride_length]
			)
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=self.num_filters,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.leaky_relu
			)
		pool2 = tf.layers.max_pooling2d(
			inputs=conv2,
			pool_size=[2, 2],
			strides=[self.stride_length, self.stride_length]
			)
		conv3 = tf.layers.conv2d(
			inputs=pool2,
			filters=self.num_filters,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.leaky_relu
			)
		pool3 = tf.layers.max_pooling2d(
			inputs=conv3,
			pool_size=[2,2],
			strides=[self.stride_length, self.stride_length]
			)


		# pool3_flat = tf.layers.Flatten()(pool3)
		# pool3_flat = tf.reshape(pool3, [self.batch_size, -1, self.conv_out_size * self.num_filters])
		pool3_flat = tf.reshape(pool3, [self.batch_size, -1, self.conv_concat])

		self.lstm_cells = [tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
		self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells,state_is_tuple=False)
		# Iteratively compute output of recurrent network
		# test = tf.placeholder(tf.float32, [self.batch_size, self.num_notes, 256])
		# outputs, self.new_state = tf.nn.dynamic_rnn(self.lstm, test, initial_state=self.init_state, dtype=tf.float32)
		self.outputs, self.new_state = tf.nn.dynamic_rnn(self.lstm, pool3_flat, dtype=tf.float32)
		# self.W_hy = tf.Variable(tf.random_normal((self.lstm_hidden_size, self.output_size),stddev=0.1),dtype=tf.float32)
		# self.b_y = tf.Variable(tf.random_normal((self.output_size,), stddev=0.1), dtype=tf.float32)
		# net_output = tf.matmul(tf.reshape(outputs, [-1, self.state_size]), self.W_hy) + self.b_y
		# net_output = tf.matmul(tf.reshape(outputs, [self.batch_size, self.state_size]), self.W_hy) + self.b_y
		# pdb.set_trace()
		# net_output = tf.matmul(tf.reshape(outputs, [-1, self.lstm_hidden_size]), self.W_hy) + self.b_y
		self.net_output = tf.layers.dense(self.outputs, self.output_size, activation=tf.nn.leaky_relu)

		self.lstm_output = tf.reshape(tf.nn.leaky_relu(self.net_output), (self.batch_size, -1, self.output_size))
		self.reshaped = tf.reshape(self.lstm_output, [self.batch_size, -1, self.conv_out_size, self.num_filters])

		self.deconv1 = tf.layers.conv2d_transpose(self.reshaped,
			filters=128,
			kernel_size=[5,5],
			padding="same",
			strides=[self.stride_length, self.stride_length],
			activation=tf.nn.relu
			)
		self.deconv2 = tf.layers.conv2d_transpose(self.deconv1,
			filters=128,
			kernel_size=[3,3],
			padding="same",
			strides=[self.stride_length, self.stride_length],
			activation=tf.nn.relu
			)
		self.final_outputs = tf.layers.conv2d_transpose(self.deconv2,
			filters=1,
			kernel_size=[3,3],
			padding="same",
			strides=[self.stride_length, self.stride_length],
			activation=tf.nn.relu
			)

		self.logits = tf.reshape(self.final_outputs, [-1, self.num_notes])
		self.labels = tf.reshape(self.y_truth, [-1, self.num_notes])

		self.losses = tf.losses.mean_squared_error(labels=self.labels, predictions=self.logits) #tf.reshape(self.y_truth, [-1, self.output_size]))
		self.total_loss = tf.reduce_mean(self.losses)
		self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.00005),0.995).minimize(self.total_loss)

	def train(self, x):
		num_timesteps = len(x[0])
		all_training = np.reshape(x, (len(x), num_timesteps, self.num_notes, 1))

		num_batches = int(len(all_training) / self.batch_size)
		all_training = all_training[:self.batch_size * num_batches]

		epoch = 0
		last_record = 0
		start_time = time.time()
		while True:
			for i in range(num_batches):
				x = all_training[i * self.batch_size : (i + 1) * self.batch_size]
				y = x
				loss, losses, prediction, _ = self.sess.run([self.total_loss, self.losses, self.logits, self.optimizer], feed_dict={self.x: x, self.y_truth: y})
				cur_record = int(time.time() - start_time / RECORD_INTERVAL)
				if cur_record > last_record:
					print("Epoch: %d - Loss: %f" % (epoch, loss))
					last_record = cur_record
					ERRORS.append(loss)

			epoch += 1
			print("Epoch: %d - Loss: %f" % (epoch, loss))

	def evaluate(self, x, length):
		# IN PROGRESS
		x = np.array([np.reshape(x, (num_timesteps, self.num_notes, 1))])
		# assume output is same size as input
		init_output, state = self.sess.run([self.final_outputs, self.lstm_last_state], {self.x: x,
			self.init_state: np.random.rand(num_timesteps, self.num_layers * 2 * self.state_size)})
		y = init_output
		last_note = np.array([init_output[0][-1]])
		for _ in range(length):
			last_note, state = self.sess.run([self.final_outputs, self.lstm_last_state], {self.x: last_note}) # need to add the init_state that is updated every loop
			y = np.append(y, last_note[0], axis=0)
		return y

def process_song(song):
	length = song.shape[0]
	num_chunks = int(length / SONG_LENGTH)
	chunks = []
	for i in range(num_chunks):
		chunks.append(np.array(song[i * SONG_LENGTH : (i+1) * SONG_LENGTH]))
	return chunks

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

model = LSTM()
training_size = 64
training, validation = data_io.test_train_sets_lpd5(ROOT_DIR)
training = list(itertools.islice(training, 0, 512))
validation = list(itertools.islice(validation, 0, 512))
x = []
for song in training:
	for clip in process_song(song):
		x.append(clip)
x = np.array(x)
model.train(x)