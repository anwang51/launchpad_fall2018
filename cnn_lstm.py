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
		self.stride_length = 2
		self.batch_size = 64 #'sentences' to look at
		self.steps = 100 #chars in a sentence ? might need to be higher, variable padding
		self.checkpoint_dir = "./checkpoint"
		self.sess = tf.Session()
		self._build_model()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def _build_model(self):
		self.x = tf.placeholder(tf.float32,[None, None, self.num_notes, 1])
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


		deconv1 = tf.layers.conv2d_transpose(pool3,
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

		self.losses = tf.losses.mean_squared_error(self.y_truth, self.final_outputs)
		self.total_loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.001),0.995).minimize(self.total_loss)



		self.lstm_last_state = np.zeros((self.num_layers * 2 * self.state_size))

		self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=net_output,labels=tf.reshape(self.y_truth, [-1, self.output_size]))
		self.total_loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.003),0.9).minimize(self.total_loss)

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
            loss, output = self.sess.run([self.total_loss, self.final_outputs], {self.x: x_mat, self.y_truth: y_mat, 
                self.init_state: np.random.rand(num_timesteps, self.num_layers * 2 * self.state_size)})
            counter +=1
            if counter %1000 == 0:
                print("Loss: %d", loss)

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
        
model = LSTM()
