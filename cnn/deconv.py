import tensorflow as tf
import skimage.io as skio
import skimage.transform as skt
import skimage.color as skco
import numpy as np
import os

class Deconv:

	def __init__(self):
		self.sess = tf.Session()
		self.batch_size = 45
		self.num_notes = 128
		self.output_size = 128
		self.stride_length = 2
		self._build_model()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, "saves/saves")

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

	def update(self, x_mat):
		return self.sess.run([self.total_loss, self.optimizer, self.final_outputs], {self.x: x_mat})

	def train(self, x_mat):
		counter = 0 
		self.training_loss = []
		while True:
			loss, _, output = self.update(x_mat)
			if counter > 0 and counter % 10 == 0:
				self.training_loss.append(loss)
				print(loss)
			counter += 1

	def evaluate(self, image):
		out = self.sess.run([self.final_outputs], {self.x: np.array([image])})
		return out[0]

folder = "/Users/wangan/Documents/launchpad_githubs/launchpad_fall2018/michelle/"

def load_images():
	imgs = []
	for filename in os.listdir(folder):
		if filename.endswith(".jpg"):
			img = skio.imread(folder + filename)
			img = skt.resize(img, (96, 128))
			img = skco.rgb2gray(img)
			img = np.expand_dims(img, 2)
			imgs.append(img)
	return imgs

train_x = load_images()
model = Deconv()

bob = skio.imread(folder + "bob.jpg")
bob = skt.resize(bob, (96, 128))
bob = skco.rgb2gray(bob)
bob = np.expand_dims(bob, 2)

y_bob = model.evaluate(bob)[0, :, :, 0]
bob = bob[:, :, 0]

sophia = skio.imread(folder + "sophia.jpg")
sophia = skt.resize(sophia, (96, 128))
sophia = skco.rgb2gray(sophia)
sophia = np.expand_dims(sophia, 2)

y_sophia = model.evaluate(sophia)[0, :, :, 0]
sophia = sophia[:, :, 0]

jonathan = skio.imread(folder + "jonathan.jpg")
jonathan = skt.resize(jonathan, (96, 128))
jonathan = skco.rgb2gray(jonathan)
jonathan = np.expand_dims(jonathan, 2)

y_jonathan = model.evaluate(jonathan)[0, :, :, 0]
jonathan = jonathan[:, :, 0]

x = train_x[30]
original = x[:, :, 0]
y = model.evaluate(x)[0, :, :, 0]

def display(img):
	skio.imshow(img)
	skio.show()