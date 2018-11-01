import tensorflow as tf
import numpy as np
import os
import random

class Music:

    def __init__(self):
        self.output_size = 128 # output size # of possible notes
		self.num_notes = 128 # input size
        self.num_layers = 2 # also for the b_lstm, can delete later
        self.batch_size = 512 # can be changed depending on training speed
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
        # Bi-directional RNN (LSTM) as the encoder
        self.x = tf.placeholder(tf.float32,[None,None, self.num_classes])
        self.y_truth = tf.placeholder(tf.float32, [None, None, self.num_classes])
        self.lstm_cells_f = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.state_size, forget_bias=1.0, state_is_tuple=False, activation=tf.nn.leaky_relu), output_keep_prob=self.global_dropout) for i in range(self.num_layers)]
        self.lstm_cells_b = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.state_size, forget_bias=1.0, state_is_tuple=False, activation=tf.nn.leaky_relu), output_keep_prob=self.global_dropout) for i in range(self.num_layers)]
        self.lstm_f = tf.contrib.rnn.MultiRNNCell(self.lstm_cells_f,state_is_tuple=False)
        self.lstm_b = tf.contrib.rnn.MultiRNNCell(self.lstm_cells_b,state_is_tuple=False)
        # Iteratively compute output of recurrent network
        outputs, (states_f, states_w) = tf.nn.bidirectional_dynamic_rnn(self.lstm_f, self.lstm_b, inputs=self.x, dtype=tf.float32)
        self.bi_final_state = tf.concat([states_f, states_w], 1)
        self.W_mu = tf.Variable(tf.random_normal((self.state_size, self.latent_dim * 2), stddev=0.01), dtype=tf.float32)
        self.b_mu = tf.Variable(tf.random_normal((self.latent_dim * 2,), stddev=0.01), dtype=tf.float32)
        self.W_sig = tf.Variable(tf.random_normal((self.state_size, self.latent_dim * 2), stddev=0.01), dtype=tf.float32)
        self.b_sig = tf.Variable(tf.random_normal((self.latent_dim * 2,), stddev=0.01), dtype=tf.float32)
        # params_mu = tf.matmul(tf.reshape(combined_output, [-1, self.state_size]), self.W_mu) + self.b_mu
        params_mu = tf.matmul(bi_final_state, self.W_mu) + self.b_mu
        params_sig = tf.matmul(bi_final_state, self.W_sig) + self.b_sig
        # borrowed from https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py
        mu = params_mu[:, self.latent_dim:]
        # The standard deviation must be positive.
        sigma =  1e-6 + tf.nn.softplus(params_sig[:, self.latent_dim:])
        # reparametrize the outputs from the encoder
        self.z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
		self.conduct_init = tf.layers.dense(self.z, self.latent_dim, activation=tf.tanh)
        # Hierarchical RNN as the decoder – 2 RNNs stacked
            # Conductor RNN - 2 layer 1024, 512 dims, vector c of lenght U (length of song), then output to shared fully-connected dense w/ tanh

            # Decoder RNN - 2 layer 1024 units per layer, output to 128 w/ softmax output layer, concat previous state like in normal rnn but also with vector c[n]

        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=net_output,labels=tf.reshape(self.y_truth, [-1, self.output_size]))
        self.total_loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.003),0.9).minimize(self.total_loss)

    def evaluate(self, letter, state):
        out, next_lstm_state = self.sess.run([self.final_outputs, self.bi_final_state],{self.x: [letter], self.init_state: [state]})
        return out[0][0], next_lstm_state

    def update(self, xbatch, ybatch):
        # init_value = np.zeros((self.batch_size, self.num_layers * 2 * self.state_size))
        return self.sess.run([self.total_loss, self.optimizer],{self.x: xbatch, self.y_truth: ybatch})

    def train(self):
        data = self.one_hotter()

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        counter = 0
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        possible_starts = range(data.shape[0] - self.steps - 1)
        x_mat = np.zeros((self.batch_size, self.steps, self.num_classes))
        y_mat = np.zeros((self.batch_size, self.steps, self.num_classes))
        batch_idxs = len(data) // self.batch_size
        EOS = np.zeros(self.num_classes)
        EOS[self.alphabets["EOS"]] = 1

        for epoch in range(20000):
            for i in range(batch_idxs):
                batch_id = random.sample(possible_starts,self.batch_size) # start idx for the batch

                for it, k in enumerate(batch_id):
                    for j in range(self.steps):
                        xs = [k + j]
                        if j == 0:
                            result = np.zeros(self.num_classes)
                            result[self.alphabets["BOS"]] = 1
                            x_mat[it, j, :] = result
                        x_mat[it, j, :] = data[xs, :]
                        if np.array_equal(data[xs], EOS):
                            break
                #
                # for j in range(self.steps):
                #     xs = [k + j for k in batch_id]
                #     ys = [k + j + 1 for k in batch_id]
                #
                #     if j == 0:
                #         result = np.zeros(32)
                #         result[self.alphabets["BOS"]] = 1
                #         x_mat[:, j, :] = result
                #     elif j == self.steps-1:
                #         result = np.zeros(32)
                #         result[self.alphabets["EOS"]] = 1
                #         x_mat[:, j, :] = result
                #     else:
                #         x_mat[:, j, :] = data[xs, :] # each 2d is a self.steps x num_classes
                #
                #     y_mat[:, j, :] = data[ys, :] # pad each 2d with a row or EOS and BOS, then concat the two outputs

                loss, opt_ = self.update(x_mat,x_mat)
                counter += 1
                if np.mod(counter, 10) == 1:  # log every 10 iters
                    print("Epoch: [%2d] [%4d/%4d], loss: %.8f" \
                        % (counter//batch_idxs, counter, batch_idxs, loss))
                if np.mod(counter, 100) == 1:  # log every 100 iters
                    print(self.talk('the'))
                if np.mod(counter, 500) == 2:
                    self.save(self.checkpoint_dir, counter)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % (self.batch_size, self.output_size)
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
        model_name = "lstm.model"
        model_dir = "%s_%s" % (self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=step)

model = Music()
