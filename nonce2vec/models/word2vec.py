"""A word2vec implementation from scratch using Tensorflow."""

from collections import defaultdict

import logging

import math
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ('Word2Vec')


class Word2Vec():
    """Tensorflow implementation of Word2vec."""

    def __init__(self, min_count, batch_size, embedding_size, num_neg_samples,
                 learning_rate, window_size, num_epochs, subsampling_rate,
                 num_threads):
        self._id2word = {}
        self._word2id = defaultdict(lambda: 0)
        self._min_count = min_count
        self._batch_size = batch_size
        self._embedding_size = embedding_size
        self._num_neg_samples = num_neg_samples
        self._learning_rate = learning_rate
        self._window_size = window_size
        self._num_epochs = num_epochs
        self._num_threads = num_threads
        self._subsampling_rate = subsampling_rate
        self._loss = None
        self._merged = None
        self._optimizer = None
        self._tf_init = None
        self._train_inputs = None
        self._train_labels = None
        self._num_batches = 0
        self._graph = None

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
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Input data.
            with tf.name_scope('inputs'):
                self._train_inputs = tf.placeholder(tf.int32,
                                                    shape=[self._batch_size])
                self._train_labels = tf.placeholder(tf.int32,
                                                    shape=[self._batch_size, 1])
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(
                    tf.random_uniform(shape=[self.vocab_size,
                                             self._embedding_size],
                                      minval=-1.0, maxval=1.0))
                embed = tf.nn.embedding_lookup(embeddings,
                                               self._train_inputs)
            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocab_size,
                                         self._embedding_size],
                                        stddev=1.0 / math.sqrt(self._embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
            with tf.name_scope('loss'):
                self._loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=self._train_labels,
                        inputs=embed,
                        num_sampled=self._num_neg_samples,
                        num_classes=self.vocab_size))
            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', self._loss)
            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                self._optimizer = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self._loss)
            self._merged = tf.summary.merge_all()
            self._tf_init = tf.global_variables_initializer()
        logger.info('Done initializing Tensorflow Graph')

    def build_vocab(self, input_filepath, output_filepath):
        """Create vocabulary-related data.

        For now assume a single file is given in input and process everything
        at once. Todo; add support for multiple files and also for processing
        as single file with multiple threads.
        """
        logger.info('Building vocabulary from file {}'.format(input_filepath))
        word_freq_dict = defaultdict(int)
        with open(input_filepath, 'r') as input_file_stream:
            for line in input_file_stream:
                for word in line.strip().split():
                    word_freq_dict[word] += 1
        word_index = 0
        self._word2id['UNK'] = word_index
        for word, freq in word_freq_dict.items():
            if freq >= self._min_count and word not in self._word2id:
                word_index += 1
                self._word2id[word] = word_index
                self._id2word[word_index] = word
        logger.info('vocabulary size = {}'.format(len(self._word2id)))
        logger.info('Saving vocabulary to file {}'.format(output_filepath))
        with open(output_filepath, 'w', encoding='utf-8') as output_stream:
            for idx, word in self._id2word.items():
                print('{}#{}'.format(idx, word), file=output_stream)

    def load_vocab(self, vocab_filepath):
        """Load a previously saved vocabulary file."""
        with open(vocab_filepath, 'r') as vocab_stream:
            for line in vocab_stream:
                (idx, word) = line.strip().split('#')
                self._word2id[word] = idx
                self._id2word[idx] = word

    def generate_batches(self, training_data_filepath, batches_filepath):
        """Generate all discretized batches in a single fileself.

        One batch per line.
        """
        logger.info('Generating batches from file {}'
                    .format(training_data_filepath))
        logger.info('Saving batches to file {}'.format(batches_filepath))
        examples = []
        with open(training_data_filepath, 'r') as training_data_stream:
            with open(batches_filepath, 'w') as batches_stream:
                for line in training_data_stream:
                    for target_id, target in enumerate(line.strip().split()):
                        for ctx_id, ctx in enumerate(line.strip().split()):
                            if ctx_id == target_id or abs(ctx_id - target_id) > self._window_size:
                                continue
                            examples.append('{}#{}'.format(self._word2id[target],
                                                           self._word2id[ctx]))
                            if len(examples) == self._batch_size:
                                print(' '.join(examples), file=batches_stream)
                                self._num_batches += 1
                                examples = []

    def _get_batch(self, batch_line):
        """Convert a stringified batch line to a valid batch."""
        batch = np.ndarray(shape=(self._batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self._batch_size, 1), dtype=np.int32)
        for idx, item in enumerate(batch_line.strip().split()):
            (ctx, target) = item.split('#')
            batch[idx] = int(target)
            labels[idx, 0] = int(ctx)
        return batch, labels

    def train(self, batches_filepath):
        """Train over the data."""
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = self._num_threads
        config.inter_op_parallelism_threads = self._num_threads
        with tf.Session(graph=self._graph, config=config) as session:
            self._tf_init.run(session=session)  # Initialize all TF variables
            average_loss = 0
            for epoch in range(1, self._num_epochs + 1):
                logger.info('epoch = {}'.format(epoch))
                with open(batches_filepath, 'r') as batches_stream:
                    for idx, batch_line in enumerate(batches_stream):
                        #progress_rate = round(((idx + 1) / self._num_batches) * 100, 1)
                        # logger.info('Epoch {}/{} | {}%'
                        #       .format(epoch, self._num_epochs, progress_rate))
                        batch_inputs, batch_labels = self._get_batch(batch_line)
                        feed_dict = {self._train_inputs: batch_inputs,
                                     self._train_labels: batch_labels}
                        _, summary, loss_val = session.run(
                            [self._optimizer, self._merged, self._loss],
                            feed_dict=feed_dict)
                        average_loss += loss_val
                        #logger.info('average loss = {}'.format(average_loss))
