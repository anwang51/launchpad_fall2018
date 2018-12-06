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
		self.num_classes = 128 # in size
		self.output_size = 128 # out size
		self.state_size = 256 # hidden size
		self.num_layers = 2
		self.batch_size = 64
		self.steps = 2048
		self.count = 0
		self.running_list_of_losses = list()
		self.test_train_seed = 1
		self.checkpoint_dir = "./checkpoint"
		self.sess = tf.Session()
		self._build_model()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def _build_model(self):
		self.x = tf.placeholder(tf.float32,[None,None, self.num_classes])
		self.y_truth = tf.placeholder(tf.float32, [None, None, self.num_classes])
		self.init_state = tf.placeholder(tf.float32, [None, self.num_layers * 2 * self.state_size])
		self.lstm_cells = [tf.nn.rnn_cell.LSTMCell(self.state_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
		self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells,state_is_tuple=False)
		# Iteratively compute output of recurrent network
		outputs, self.new_state = tf.nn.dynamic_rnn(self.lstm, self.x, initial_state=self.init_state, dtype=tf.float32)
		self.W_hy = tf.Variable(tf.random_normal((self.state_size, self.num_classes),stddev=0.01),dtype=tf.float32)
		self.b_y = tf.Variable(tf.random_normal((self.output_size,), stddev=0.01), dtype=tf.float32)
		net_output = tf.matmul(tf.reshape(outputs, [-1, self.state_size]), self.W_hy) + self.b_y

		self.final_outputs = tf.reshape(tf.nn.softmax(net_output),(tf.shape(outputs)[0], tf.shape(outputs)[1], self.output_size))
		self.lstm_last_state = np.zeros((self.num_layers * 2 * self.state_size))
		self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=net_output,labels=tf.reshape(self.y_truth, [-1, self.output_size]))
		self.total_loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.003),0.9).minimize(self.total_loss)


	def train(self, x):
		num_timesteps = len(x[0])
		all_training = np.reshape(x, (len(x), num_timesteps, self.num_notes))

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