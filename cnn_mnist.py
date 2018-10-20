from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 
import pdb

tf.logging.set_verbosity(tf.logging.INFO)


class CNN:

    def __init__(self):
        self.height = 28 # Image height
        self.width = 28 # Image width
        self.channels = 1 # Image channels
        self.classes = 10 # Number of classifications
        self.loss = None
        self.optimizer = None
        self.x = None
        self.y_truth = None
        self.training = False
        self.sess = tf.Session()
        self.training_loss = []
        self.build_model()
        self.sess.run(tf.global_variables_initializer())


    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels])
        self.y_truth = tf.placeholder(tf.int32, [None, 1])
        conv1 = tf.layers.conv2d(
            inputs=self.x,
            filters=32,
            kernel_size=[3,3],
            padding="same",
            activation=tf.nn.relu
            )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2,2],
            strides=[2, 2]
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
            strides=[2,2]
            )
        pool2_flat = tf.reshape(pool2, [-1, int(self.width / (2*2) * self.height / (2*2) * 64)])
        dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu
            )
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=self.training
            )
        self.logits = tf.layers.dense(inputs=dropout, units=self.classes)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y_truth, logits=self.logits)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(
                loss=self.loss,
                global_step=tf.train.get_global_step()
                )

    def update(self, x_mat, y_mat):
        return self.sess.run([self.loss, self.optimizer, self.logits], {self.x: x_mat, self.y_truth: y_mat})

    def train(self):
        self.training = True
        x_mat, y_mat = self.get_training_data()
        counter = 0 
        while True:
            counter += 1
            x = np.array([np.reshape(
                        x_mat[counter], 
                        (self.height, self.width, self.channels))])
            y = np.array([[y_mat[counter]]])
            loss, _, _ = self.update(x, y)
            if counter > 0 and counter % 1000 == 0:
                self.training_loss.append(loss)
                print(loss)

    def evaluate(self, x):
        self.training = False
        logits = self.sess.run(self.logits, {self.x: x})
        return logits, max(range(len(logits)), key=lambda i:logits[i])

    def get_training_data(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        return train_data, train_labels

if __name__ == "__main__":
    cnn = CNN()
