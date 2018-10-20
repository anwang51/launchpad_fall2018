import tensorflow as tf

class Deconv:

	def __init__():
		self.sess = tf.Session()
		self._build_model()
		self.sess.run(tf.global_variables_initializer())

	def _build_model(self):
		self.x = tf.placeholder(tf.float32,[None, None, self.num_notes, 1])
		self.y_truth = tf.placeholder(tf.float32, [None, None, self.num_notes, 1])

		conv1 = tf.layers.conv2d(
			inputs=self.x,
			filters=32,
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
			filters=64,
			kernel_size=[5,5],
			padding="same",
			activation=tf.nn.relu
			)
		pool2 = tf.layers.max_pooling2d(
			inputs=conv2,
			pool_size=[2,2],
			strides=[self.stride_length, self.stride_length]
			)
		pool2_flat = tf.layers.Flatten()(pool2)
		pool2_flat = tf.expand_dims(pool2_flat, 2)

		reshaped = tf.reshape(pool2_flat, [-1, self.num_notes/(self.stride_length * self.stride_length), -1, 1])
		
		deconv1 = tf.nn.conv2d_transpose(reshaped,
			tf.placeholder(tf.float32, shape=[3, 3, 1, 1]),
			tf.stack([tf.shape(self.x)[0]/2, tf.shape(self.x)[1]/2, 1, 1]),
			[1, 2, 2, 1],
			padding="SAME")
		self.final_outputs = tf.nn.conv2d_transpose(deconv1,
			tf.placeholder(tf.float32, shape=[3, 3, 1, 1]),
			tf.stack([tf.shape(self.x)[0], tf.shape(self.x)[1], 1, 1]),
			[1, 2, 2, 1],
			padding="SAME")
		self.lstm_last_state = np.zeros((self.num_layers * 2 * self.state_size))

		self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=net_output,labels=tf.reshape(self.y_truth, [-1, self.output_size]))
		self.total_loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.003),0.9).minimize(self.total_loss)

	def update():

	def train():

	def evaluate():
		