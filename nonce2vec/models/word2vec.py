"""A word2vec implementation from scratch using Tensorflow."""

from collections import defaultdict

import math
import six
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def _generate_batch():
    pass


class Word2Vec():

    def __init__(self, min_count, batch_size, embedding_size, num_neg_samples,
                 learning_rate, window_size):
        self._word_freq = defaultdict(int)
        self._id2word = {}
        self._word2id = {}
        self._discretized_data = []
        self._min_count = min_count
        self._batch_size = batch_size
        self._embedding_size = embedding_size
        self._num_neg_samples = num_neg_samples
        self._learning_rate = learning_rate
        self._window_size = window_size
        self._loss = None
        self._optimizer = None
        self._tf_init = None
        self._train_inputs = None
        self._train_labels = None

    @property
    def vocab_size(self):
        """Return vocabulary length."""
        return len(self._word2id)

    def initialize_tf_graph(self):
        """Initialize the Tensorflow Graph with all the necessary variables."""
        if not self.vocab_size:
            raise Exception('You need to initialize the vocabulary with '
                            'build_vocab before initializing the Tensorflow '
                            'Graph')
        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            with tf.name_scope('inputs'):
                self._train_inputs = tf.placeholder(tf.int32,
                                                    shape=[self._batch_size])
                self._train_labels = tf.placeholder(tf.int32,
                                                    shape=[self._batch_size, 1])
            # Ops and variables pinned to the CPU because of
            # missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([self.vocab_size,
                                           self._embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocab_size,
                                         self._embedding_size],
                                        stddev=1.0 / math.sqrt(self._embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels
            # each time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            # http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                self._loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=self._num_neg_samples,
                        num_classes=self.vocab_size))
            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                self._optimizer = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self._loss)
        self._tf_init = tf.global_variables_initializer()
        print('Done initializing Tensorflow Graph')

    def build_vocab(self, input_filepath):
        """Create vocabulary-related data.

        For now assume a single file is given in input and process everything
        at once. Todo; add support for multiple files and also for processing
        as single file with multiple threads.
        """
        print('Building vocabulary from file {}'.format(input_filepath))
        with open(input_filepath, 'r') as input_file_stream:
            for line in input_file_stream:
                for word in line.strip().split():
                    self._word_freq[word] += 1
        word_index = 0
        self._word2id['UNK'] = word_index
        for word, freq in self._word_freq.items():
            if freq > self._min_count and word not in self._word2id:
                word_index += 1
                self._word2id[word] = word_index
                self._id2word[word_index] = word
        print('vocabulary size = {}'.format(len(self._word2id)))


    def _generate_batch(self):
        pass

    def train(self):
        with tf.Session(graph=self._graph) as session:
            self._init.run()  # Initialize all TF variables
            average_loss = 0
            for step in six.moves.xrange(self._num_steps):
                batch_inputs, batch_labels = self._generate_batch()
                feed_dict = {self._train_inputs: batch_inputs,
                             self._train_labels: batch_labels}
                # We perform one update step by evaluating the optimizer op
                # (including it in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the
                # returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph
                # in TensorBoard.
                _, summary, loss_val = session.run(
                    [self._optimizer, self._merged, self._loss],
                    feed_dict=feed_dict)
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0


if __name__ == '__main__':
    INPUT_FILE = '/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.test'
    #INPUT_FILE = '/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.all.utf8.sent.split.lower'
    w2v = Word2Vec(min_count=1, batch_size=128, embedding_size=128,
                   num_neg_samples=64, learning_rate=1.0, window_size=15)
    w2v.build_vocab(INPUT_FILE)
    w2v.preprocess(INPUT_FILE)
    w2v.initialize_tf_graph()
    #w2v.train()
