import tensorflow as tf
import numpy as np
import os
import sys
import random
import pdb
import time

import itertools
import data_io

ROOT_DIR = "lpd_5_cleansed/"
RECORD_INTERVAL = 300 # 900 seconds
ERRORS = []
SONG_LENGTH = 512

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
        self.path_to_data = "/Users/wangan/Documents/launchpad_githubs/launchpad_fall2018/lstm/lpd_5_cleansed"
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
        self.losses = tf.losses.mean_squared_error(predictions=net_output,labels=tf.reshape(self.y_truth, [-1, self.output_size]))
        self.total_loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.005),0.995).minimize(self.total_loss)

    def evaluate(self, letter=None, state=None):
        if state is None:
            state = np.zeros((self.num_layers * 2 * self.state_size))
        if letter is None:
            letter = np.zeros((1, self.num_classes))
            letter[0][0] = 1
        out, next_lstm_state = self.sess.run([self.final_outputs, self.new_state],{self.x: [letter], self.init_state: [state]})
        return out[0][0], next_lstm_state[0]

    def update(self, xbatch, ybatch):
        init_value = np.zeros((self.batch_size, self.num_layers * 2 * self.state_size))
        return self.sess.run([self.total_loss, self.optimizer],{self.x: xbatch, self.y_truth: ybatch, self.init_state: init_value})

    def train(self, x):
        num_timesteps = len(x[0])
        all_training = x

        num_batches = int(len(all_training) / self.batch_size)
        all_training = all_training[:self.batch_size * num_batches]

        epoch = 0
        last_record = 0
        start_time = time.time()
        while True:
            for i in range(num_batches):
                x = all_training[i * self.batch_size : (i + 1) * self.batch_size]
                y = x
                loss, opt_ = self.update(x, y)
                cur_record = int(time.time() - start_time / RECORD_INTERVAL)
                if cur_record > last_record:
                    print("Epoch: %d - Loss: %f" % (epoch, loss))
                    last_record = cur_record
                    ERRORS.append(loss)

            epoch += 1
            print("Epoch: %d - Loss: %f" % (epoch, loss))

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
    
    def save_checkpoint():
        self.save(self.checkpoint_dir, self.count)

    def save(self, checkpoint_dir, step):
        model_name = "lstm.model"
        model_dir = "%s_%s" % (self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=step)

def process_song(song):
    length = song.shape[0]
    num_chunks = int(length / SONG_LENGTH)
    chunks = []
    for i in range(num_chunks):
        chunks.append(np.array(song[i * SONG_LENGTH : (i+1) * SONG_LENGTH]))
    return chunks

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