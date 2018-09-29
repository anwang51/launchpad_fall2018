import tensorflow as tf
import numpy as np
import os
import random

# does not do end-of-sentence

class LSTM:

    def __init__(self):
        self.num_classes = 32 #in size
        self.output_size = 32 #out size
        self.state_size = 256 #hidden size
        self.num_layers = 2
        self.batch_size = 64 #'sentences' to look at at once
        self.steps = 100 #chars in a sentence ? might need to be higher, variable padding
        self.checkpoint_dir = "./checkpoint"
        self.sess = tf.Session()
        self._build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.alphabets = {'a' : 0, 'b': 1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z': 25, '.': 26, '!':27, '?':28, ' ':29, ',':30, '%':31}

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

    def one_hotter(self):
        data_ = ""
        with open('shakespeare.txt', 'r') as f:
            data_ += f.read()
        data_ = data_.lower()
        letter_vectors = []
        for l in data_:
            if l in self.alphabets:
                result = np.zeros(32)
                result[self.alphabets[l]] = 1
                letter_vectors.append(result)
            elif l in ['.','?','!']:
                letter_vectors.append(self.alphabets["%"])
        return np.array(letter_vectors)
    	# return tf.one_hot(alphabets[letter], 32, on_value=1, off_value=0).eval(session=self.sess)

    def evaluate(self, letter, state):
        out, next_lstm_state = self.sess.run([self.final_outputs, self.new_state],{self.x: [letter], self.init_state: [state]})
        return out[0][0], next_lstm_state[0]

    def update(self, xbatch, ybatch):
        init_value = np.zeros((self.batch_size, self.num_layers * 2 * self.state_size))
        return self.sess.run([self.total_loss, self.optimizer],{self.x: xbatch, self.y_truth: ybatch, self.init_state: init_value})

    def talk(self):
        test_word = 'the'
        state = np.zeros((self.num_layers * 2 * self.state_size))
        for i in range(3):
            letter_vector = np.zeros((1,32))
            letter_vector[0,self.alphabets[test_word[i]]] = 1
            out, state = self.evaluate(letter_vector,state)
        for i in range(500):
            element = np.random.choice(range(self.num_classes), p=out)
            test_word += list(self.alphabets.keys())[list(self.alphabets.values()).index(element)]
            letter_vector = np.zeros((1,32))
            letter_vector[0,element] = 1
            out, state = self.evaluate(letter_vector,state)
        return test_word

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

        for epoch in range(20000):
            for i in range(batch_idxs):
                batch_id = random.sample(possible_starts,self.batch_size) # start idx for the batch

                for j in range(self.steps):
                    xs = [k + j for k in batch_id]
                    ys = [k + j + 1 for k in batch_id]

                    x_mat[:, j, :] = data[xs, :] # each 2d is a self.steps x num_classes
                    y_mat[:, j, :] = data[ys, :]

                loss, opt_ = self.update(x_mat,y_mat)
                counter += 1
                if np.mod(counter, 10) == 1:  # log every 10 iters
                    print("Epoch: [%2d] [%4d/%4d], loss: %.8f" \
                        % (epoch, i+1, batch_idxs, loss))
                if np.mod(counter, 100) == 1:  # log every 100 iters
                    print(self.talk())
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

model = LSTM()
