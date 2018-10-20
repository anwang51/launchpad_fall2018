import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist

np.random.seed(0)
tf.set_random_seed(0)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

class VAE(object):
    """A Variational Autoencoder Network comprising two encoding layers and
    two decoding layers, sampling from latent variable distributions.

    Inspired by https://jmetzen.github.io/2015-11-27/vae.html
    """
    def __init__(self):
        # Initialize self
        self.input_size = 784
        self.e1_size = 500
        self.e2_size = 500
        self.z_size = 20
        self.d1_size = 500
        self.d2_size = 500
        self.learning_rate = 0.001
        self.batch_size = 100
        self.act_fn = tf.nn.relu

        self.build_network()
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def build_network(self):
        # Build VAE network: encoding layers, sampling, decoding/generation layers
        self.x = tf.placeholder(tf.float32, [None, input_size])
        e1_layer = tf.nn.dense(self.x, self.e1_size, activation=self.act_fn)
        e2_layer = tf.nn.dense(e1_layer, self.e2_size, activation=self.act_fn)
        latent_encoding = tf.nn.dense(e2_layer, self.z_size, activation=self.act_fn)

        # LADKJFSK some code to sample from latent_encoding distribution

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((None, self.z_size), 0, 1,
                    dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))


        d1_layer = tf.nn.dense(some_sample, self.d1_size, activation=self.act_fn)
        d2_layer = tf.nn.dense(d1_layer, self.d2_size, activation=self.act_fn)
        self.output = tf.nn.dense(d2_layer, self.input_size, activation=self.act_fn)

    def init_weights(self):
        # Setup weights

    def encoder_network(self):
        # Encoding section

    def decoder_network(self):
        # Decoder/generation section

    def create_loss_optimizer(self):
        """Build a loss function from reconstruction and latent loss quantities.

            1) The reconstruction loss (the negative log probability
            of the input under the reconstructed Bernoulli distribution
            induced by the decoder in the data space).
            This can be interpreted as the number of "nats" required
            for reconstructing the input when the activation in latent
            is given. (Adding 1e-10 to avoid evaluation of log(0)).

            2) The latent loss, which is defined as the Kullback Leibler divergence
            between the distribution in latent space induced by the encoder on
            the data and some prior. This acts as a kind of regularizer.
            This can be interpreted as the number of "nats" required
            for transmitting the the latent space distribution given
            the prior.
        """
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, batch):
        # Fit VAE parameters to a batch

    def train(self, epochs):
        for epoch in range(epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = mnist.train.next_batch(batch_size)

                # Fit training using batch data
                cost = vae.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(avg_cost))
