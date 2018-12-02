import tensorflow as tf
import numpy as np
import os
import scipy.misc

class vae_mnist():
    def __init__(self):
        self.x_mat, self.label = self.get_training_data()
        self.n_samples = len(self.x_mat)

        self.n_hidden = 500
        self.n_z = 2
        self.batch_size = 100
        self.checkpoint_dir = './vae_mnist'
        self.model_dir = "%s_%s" % (self.batch_size, self.n_z)
        self.sess = tf.Session()
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, 784])
        # image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        self.mu, sigma = self.encoder(self.images)

        # reparametrize the outputs from the encoder
        z = self.mu + sigma * tf.random_normal([self.batch_size,self.n_z], 0, 1, dtype=tf.float32)

        #decoder
        self.generated = self.decoder(z)
        # print(self.generated.get_shape())

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-10 + self.generated) + (1-self.images) * tf.log(1e-10 + 1 - self.generated),1) #marginal_likelihood
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    # encoder
    def encoder(self, x):
        # w0 = tf.get_variable('w0', [28, self.n_hidden], initializer=w_init)
        w0 = tf.Variable(tf.random_normal((x.get_shape().as_list()[1], self.n_hidden), stddev=0.01), dtype=tf.float32) # equivalent?
        b0 = tf.Variable(tf.random_normal((self.n_hidden,), stddev=0.01), dtype=tf.float32)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.leaky_relu(h0)
        # can add dropout

        w1 = tf.Variable(tf.random_normal((h0.get_shape().as_list()[1], self.n_hidden), stddev=0.01), dtype=tf.float32)
        b1 = tf.Variable(tf.random_normal((self.n_hidden,), stddev=0.01), dtype=tf.float32)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.leaky_relu(h1)

        w_out = tf.Variable(tf.random_normal((h1.get_shape().as_list()[1], self.n_z * 2), stddev=0.01), dtype=tf.float32)
        b_out = tf.Variable(tf.random_normal((self.n_z * 2,), stddev=0.01), dtype=tf.float32)
        params = tf.matmul(h1, w_out) + b_out

        mu = params[:, :self.n_z]
        # The standard deviation must be positive.
        sigma =  1e-6 + tf.nn.softplus(params[:, self.n_z:])

        return mu, sigma

    def decoder(self, z):
        w0 = tf.Variable(tf.random_normal((z.get_shape().as_list()[1], self.n_hidden), stddev=0.01), dtype=tf.float32)
        b0 = tf.Variable(tf.random_normal((self.n_hidden,), stddev=0.01), dtype=tf.float32)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.leaky_relu(h0)
        # can add dropout

        w1 = tf.Variable(tf.random_normal((h0.get_shape().as_list()[1], self.n_hidden), stddev=0.01), dtype=tf.float32)
        b1 = tf.Variable(tf.random_normal((self.n_hidden,), stddev=0.01), dtype=tf.float32)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.leaky_relu(h1)

        w_out = tf.Variable(tf.random_normal((h1.get_shape().as_list()[1], 784), stddev=0.01), dtype=tf.float32)
        b_out = tf.Variable(tf.random_normal((784,), stddev=0.01), dtype=tf.float32)
        y = tf.sigmoid(tf.matmul(h1, w_out) + b_out)

        return y

    def train(self):
        self.training = True
        counter = 0
        validator = np.copy(self.x_mat[:self.batch_size])
        reshape_validator = validator.reshape(self.batch_size,28,28)
        v_labels = np.copy(self.label[:self.batch_size])
        np.save('vae_mnist/v_labels.npy', v_labels)
        scipy.misc.imsave("vae_mnist/base.jpg",self.merge(reshape_validator[:64],[8,8]))

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        total_batch = self.n_samples // self.batch_size

        for epoch in range(2000):
            np.random.shuffle(self.x_mat)
            for i in range(total_batch):
                counter += 1
                offset = (i * self.batch_size) % (self.n_samples)
                batch_xs_input = self.x_mat[offset:(offset + self.batch_size), :]

                _, gen_loss, lat_loss = self.sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch_xs_input})

            print("epoch %d: gen_loss %f lat_loss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
            self.save(self.checkpoint_dir, counter)
            generator_test = self.sess.run(self.generated, feed_dict={self.images: validator})
            generator_test = generator_test.reshape(self.batch_size,28,28)
            scipy.misc.imsave(os.path.join(self.checkpoint_dir, self.model_dir)+'/'+str(counter//550)+'.jpg', self.merge(generator_test[:64], [8,8]))
            if self.n_z == 2:
                z_mu = self.sess.run(self.mu, feed_dict={self.images: validator})
                np.save('vae_mnist/mu.npy', z_mu)
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(8, 6))
            # plt.scatter(z_mu[:, 0], z_mu[:, 1], c=y_sample)
            # plt.colorbar()
            # plt.grid()
            # plt.show()
            # z_mu = np.load('vae_mnist/mu.npy')
            # y_sample = np.load('vae_mnist/v_labels.npy')

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % (self.batch_size, self.n_z)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            import re
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save(self, checkpoint_dir, step):
        model_name = "vae.model"
        model_dir = "%s_%s" % (self.batch_size, self.n_z)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=step)

    def merge(self, images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w] = image
        return img

    def get_training_data(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        return train_data, train_labels


model = vae_mnist()
# model.train()
