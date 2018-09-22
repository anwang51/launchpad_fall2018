import numpy as np

import backend
import nn


class Model(object):
    """Base model class for the different applications"""

    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class DeepQModel(Model):
    """
    TODO: Question 7 - [Application] Reinforcement Learning

    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.m1 = nn.Variable(4, 100)
        self.b1 = nn.Variable(100)
        self.m2 = nn.Variable(100, 2)
        self.b2 = nn.Variable(2)

    def run(self, states, Q_target=None):
        """
        TODO: Question 7 - [Application] Reinforcement Learning

        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2])
        input_x = nn.Input(graph, states)
        layer1 = nn.ReLU(graph, nn.MatrixVectorAdd(graph, nn.MatrixMultiply(graph, input_x, self.m1), self.b1))
        layer2 = nn.MatrixVectorAdd(graph, nn.MatrixMultiply(graph, layer1, self.m2), self.b2)

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, Q_target)
            loss = nn.SquareLoss(graph, layer2, input_y)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(layer2)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    TODO: Question 8 - [Application] Language Identification

    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.05
        self.decay_rate = 0.9997
        # self.decay_rate = 1.0
        self.bushiness = 128
        self.h_size = 128
        self.m1 = nn.Variable(self.num_chars, self.bushiness)
        self.m2 = nn.Variable(self.h_size, self.bushiness)
        self.dense1 = nn.Variable(self.bushiness, self.bushiness)
        self.d1 = nn.Variable(self.bushiness)
        self.dense2 = nn.Variable(self.bushiness, self.bushiness)
        self.d2 = nn.Variable(self.bushiness)
        self.dense3 = nn.Variable(self.bushiness, self.h_size)
        self.d3 = nn.Variable(self.h_size)
        self.dense4 = nn.Variable(self.h_size, self.bushiness)
        self.d4 = nn.Variable(self.bushiness)
        self.dense5 = nn.Variable(self.bushiness, len(self.languages))
        self.d5 = nn.Variable(len(self.languages))

    def run(self, xs, y=None):
        """
        TODO: Question 8 - [Application] Language Identification

        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"
        self.learning_rate *= self.decay_rate
        h = np.zeros((batch_size, self.h_size))
        graph = nn.Graph([self.m1, self.m2, self.dense1, self.d1, self.dense2, self.d2, self.dense3, self.d3,
                self.dense4, self.d4, self.dense5, self.d5])
        h_out = nn.Input(graph, h)
        for x in xs:
            input_x = nn.Input(graph, x)
            layer1 = nn.ReLU(graph, nn.Add(graph, nn.MatrixMultiply(graph, input_x, self.m1), nn.MatrixMultiply(graph, h_out, self.m2)))
            dense1 = nn.ReLU(graph, nn.MatrixVectorAdd(graph, nn.MatrixMultiply(graph, layer1, self.dense1), self.d1))
            # dense2 = nn.ReLU(graph, nn.MatrixVectorAdd(graph, nn.MatrixMultiply(graph, dense1, self.dense2), self.d2))
            h_out = nn.MatrixVectorAdd(graph, nn.MatrixMultiply(graph, dense1, self.dense3), self.d3)
            # h = graph.get_output(dense3)
        layer4 = nn.ReLU(graph, nn.MatrixVectorAdd(graph, nn.MatrixMultiply(graph, h_out, self.dense4), self.d4))
        final = nn.MatrixVectorAdd(graph, nn.MatrixMultiply(graph, layer4, self.dense5), self.d5)

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, final, input_y)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(final)
