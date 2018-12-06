import tensorflow as tf
import numpy as np
import os
import random
from data import data_io
from data import midi_proc
import itertools

SPLIT_LEN = 128
training_size = 64

class Music:

    def __init__(self):
        self.n_samples = training_size
        self.output_size = 128 # output size # of possible notes
        self.num_notes = 128 # input size
        self.num_layers = 2 # also for the b_lstm, can delete later
        self.batch_size = 64 # can be changed depending on training speed
        self.state_size = 2048 # hidden layer of features for the blstm
        # implement weight decay
        self.decay_rate = .9999
        self.learning_rate = 0.001
        self.latent_dim = 512
        self.global_dropout = 0.6 # not final
        self.checkpoint_dir = "./music_checkpoint"
        self.sess = tf.Session()
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_model(self):
        with tf.device('/gpu:0'):
            # Bi-directional RNN (LSTM) as the encoder
            self.x = tf.placeholder(tf.float32,[None,None, self.num_notes]) # any number of songs for the batch

            self.mu, sigma = self.encoder(self.x)

            # reparametrize the outputs from the encoder
            self.z = self.mu + sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

            # Hierarchical RNN as the decoder – 2 RNNs stacked – (also use seq2seq for attention?)
            # self.conduct = self.conductor(self.z)
            self.init_state = tf.placeholder(tf.float32, [None, self.num_layers * 2 * 1024])
            net_output, self.state = self.decoder(self.z,self.init_state)

            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=net_output,labels=tf.reshape(self.x, [-1, self.output_size]))
            self.total_loss = tf.reduce_mean(self.losses)

            # self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.003),0.9).minimize(self.total_loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta2=0.9999).minimize(self.total_loss)

    def encoder(self, x):
        with tf.device('/gpu:0'):
            # self.y_truth = tf.placeholder(tf.float32, [None, None, self.num_classes])
            self.lstm_cells_f = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.state_size, forget_bias=1.0, state_is_tuple=False, activation=tf.nn.leaky_relu), output_keep_prob=self.global_dropout) for i in range(self.num_layers)]
            self.lstm_cells_b = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.state_size, forget_bias=1.0, state_is_tuple=False, activation=tf.nn.leaky_relu), output_keep_prob=self.global_dropout) for i in range(self.num_layers)]
            self.lstm_f = tf.contrib.rnn.MultiRNNCell(self.lstm_cells_f,state_is_tuple=False)
            self.lstm_b = tf.contrib.rnn.MultiRNNCell(self.lstm_cells_b,state_is_tuple=False)
            # Iteratively compute output of recurrent network
            outputs, (states_f, states_w) = tf.nn.bidirectional_dynamic_rnn(self.lstm_f, self.lstm_b, inputs=x, dtype=tf.float32)
            bi_final_state = tf.concat([states_f, states_w], 1)
            self.W_mu = tf.Variable(tf.random_normal((16384, self.latent_dim * 2), stddev=0.01), dtype=tf.float32)
            self.b_mu = tf.Variable(tf.random_normal((self.latent_dim * 2,), stddev=0.01), dtype=tf.float32)
            self.W_sig = tf.Variable(tf.random_normal((16384, self.latent_dim * 2), stddev=0.01), dtype=tf.float32)
            self.b_sig = tf.Variable(tf.random_normal((self.latent_dim * 2,), stddev=0.01), dtype=tf.float32)
            # params_mu = tf.matmul(tf.reshape(combined_output, [-1, self.state_size]), self.W_mu) + self.b_mu
            params_mu = tf.matmul(bi_final_state, self.W_mu) + self.b_mu
            params_sig = tf.matmul(bi_final_state, self.W_sig) + self.b_sig
            # borrowed from https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py
            mu = params_mu[:, :self.latent_dim]
            # The standard deviation must be positive.
            sigma = 1e-6 + tf.nn.softplus(params_sig[:, self.latent_dim:])

            return mu, sigma

    # def conductor(self, z):
    #     hidden_size = 1024
    #     output_size = 512
    #     conduct_init = tf.layers.dense(z, self.latent_dim, activation=tf.tanh)
    #     # Conductor RNN - 2 layer 1024, 512 dims, vector c of lenght U (length of song), then output to shared fully-connected dense w/ tanh
    #     init_state = tf.placeholder(tf.float32, [None, self.num_layers * 2 * hidden_size])
    #     lstm_cells = [tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
    #     lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells,state_is_tuple=False)
    #     print(conduct_init.shape, z.shape)
    #     # with tf.variable_scope('conductor'):
    #     outputs, new_state = tf.nn.dynamic_rnn(lstm, tf.expand_dims(conduct_init, axis = 2), initial_state=init_state, dtype=tf.float32)
    #     # outputs, new_state = tf.nn.static_rnn([lstm], conduct_init, initial_state=init_state, dtype=tf.float32)
    #     W_hy = tf.Variable(tf.random_normal((hidden_size, output_size),stddev=0.01),dtype=tf.float32)
    #     b_y = tf.Variable(tf.random_normal((output_size,), stddev=0.01), dtype=tf.float32)
    #     net_output = tf.matmul(tf.reshape(outputs, [-1, hidden_size]), W_hy) + b_y
    #
    #     final_outputs = tf.reshape(tf.nn.softmax(net_output),(tf.shape(outputs)[0], tf.shape(outputs)[1], output_size)) # dont know if this line is supposed to be here
    #     return final_outputs

    def decoder(self, c, init_state):
        with tf.device('/gpu:0'):
            # Decoder RNN - 2 layer 1024 units per layer, output to 128 w/ softmax output layer, concat previous state like in normal rnn but also with vector c[n]
            # will output like how the basic rnn does, just the first ouput and state, run this portion autoregressively when evaluating.
            conduct_init = tf.layers.dense(c, self.output_size, activation=tf.tanh)
            conduct_init = tf.expand_dims(conduct_init, axis = 2)
            lstm_cells = [tf.nn.rnn_cell.LSTMCell(1024, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
            # lstm_cells = tf.contrib.rnn.LSTMCell(1024, forget_bias=1.0, state_is_tuple=False)
            lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells,state_is_tuple=False)
            # Iteratively compute output of recurrent network
            # with tf.variable_scope('decoder'): # maybe just make it one layer
                # outputs, self.new_state = tf.nn.dynamic_rnn(lstm, conduct_init, initial_state=init_state, dtype=tf.float32)
            outputs, self.new_state = tf.nn.dynamic_rnn(lstm, conduct_init, initial_state=init_state, dtype=tf.float32)
            W_hy = tf.Variable(tf.random_normal((1024, 128),stddev=0.01),dtype=tf.float32)
            b_y = tf.Variable(tf.random_normal((128,), stddev=0.01), dtype=tf.float32)
            net_output = tf.matmul(tf.reshape(outputs, [-1, 1024]), W_hy) + b_y

            self.final_outputs = tf.reshape(tf.nn.softmax(net_output),(tf.shape(outputs)[0], tf.shape(outputs)[1], 128))

            return self.final_outputs, self.new_state #use these when regressively calling the decoder rnn

    # def evaluate(self, letter, state):
    #     out, next_lstm_state = self.sess.run([self.final_outputs, self.bi_final_state],{self.x: [letter], self.init_state: [state]})
    #     return out[0][0], next_lstm_state

    # def update(self, xbatch, ybatch):
    #     # init_value = np.zeros((self.batch_size, self.num_layers * 2 * self.state_size))
    #     return self.sess.run([self.total_loss, self.optimizer],{self.x: xbatch, self.y_truth: ybatch})

    def train(self, xmat):
        self.training = True
        counter = 0
        # VALIDATION AND SCIPY DISPLAY FUNCTIONS COMMENTED OUT FOR NOW

        # validator = np.copy(np.array(list(itertools.islice(self.x_mat, self.batch_size))))
        # reshape_validator = validator.reshape(self.batch_size,28,28)
        # v_labels = np.copy(self.label[:self.batch_size])
        # np.save('vae_mnist/v_labels.npy', v_labels)
        # scipy.misc.imsave("vae_mnist/base.jpg",self.merge(reshape_validator[:64],[8,8]))

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        total_batch = self.n_samples // self.batch_size

        for epoch in itertools.count():
            # np.random.shuffle(self.x_mat)
            x_iter = self.grouper(self.batch_size, xmat) # batches of input songs
            for i in range(total_batch):
                counter += 1
                offset = (i * self.batch_size) % (self.n_samples)
                self.batch_xs_input = np.array(next(x_iter)).reshape(self.batch_size, SPLIT_LEN)
                self.batch_xs_input = [self.one_hot(b) for b in self.batch_xs_input]
                init_value = np.zeros((self.batch_size, self.num_layers * 2 * 1024))
                _, gen_loss, tot_loss = self.sess.run((self.optimizer, self.losses, self.total_loss), feed_dict={self.x: self.batch_xs_input, self.init_state: init_value})

            print("epoch %d: gen_loss %f lat_loss %f" % (epoch, np.mean(gen_loss), np.mean(tot_loss)))
            # self.save(self.checkpoint_dir, counter)
            # generator_test = self.sess.run(self.generated, feed_dict={self.images: validator})
            # generator_test = generator_test.reshape(self.batch_size,28,28)
            # scipy.misc.imsave(os.path.join(self.checkpoint_dir, self.model_dir)+'/'+str(counter//550)+'.jpg', self.merge(generator_test[:64], [8,8]))
            # if self.n_z == 2:
            #     z_mu = self.sess.run(self.mu, feed_dict={self.images: validator})
            #     np.save('vae_mnist/mu.npy', z_mu)

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
        model_dir = "%s_%s" % (self.batch_size, self.num_layers)
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
        model_dir = "%s_%s" % (self.batch_size, self.num_layers)
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
        # OLD MNIST TRAINING DATA CODE
        # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        # train_data = mnist.train.images
        # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        # eval_data = mnist.test.images
        # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        # return train_data, train_labels

        # NEW TRAINING DATA FROM LPD_5
        return itertools.islice(train, training_size)

    def grouper(self, n, iterable): # chunks and returns iterators of size n from iterable (batches)
        it = iter(iterable)
        while True:
           chunk = tuple(itertools.islice(it, n))
           if not chunk:
               return
           yield chunk

    def one_hot(self,b):
        letter_vectors = []
        for i in b:
            result = np.zeros(128)
            result[i] = 1
            letter_vectors.append(result)
        return letter_vectors

test, train = data_io.test_train_sets_lpd5("./lpd_5_cleansed", track_name='Piano', split_len=SPLIT_LEN)

def train_yielder():
    for t in train:
        mono = midi_proc.convert_to_mono(t)
        # print(mono.size)
        yield mono

train = list(itertools.islice(train_yielder(), training_size))
model = Music()
model.train(train)
