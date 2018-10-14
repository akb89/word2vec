"""A word2vec implementation from scratch using Tensorflow."""

import os

from collections import defaultdict

import logging
import time

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
        self._embeddings = None
        self._vectors = None
        self._train_inputs = None
        self._train_labels = None
        self._graph = None
        self._saver = None
        self._timers = {
            'batches_generation': None,
            'training': None,
            'lines_reading': None,
        }

        self._timings = {
            'batches_generation': 0.,
            'training': 0.,
            'lines_reading': 0.
        }

    @property
    def normalized_embeddings(self):
        if not self._vectors:
            # TODO: check: is this L2 euclidian norm?
            l2_norm = tf.sqrt(tf.reduce_sum(tf.square(self._embeddings), 1, keepdims=True))
            normalized_embeddings = self._embeddings / l2_norm
            self._vectors = normalized_embeddings.eval()
        return self._vectors

    @property
    def vector(self, word):
        """Get the vector corresponding to a word."""
        return self._vectors[self._word2id[word]]

    @property
    def cosine_sim(self, word_1, word_2):
        """Get the cosine between two words."""
        pass

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
            with tf.name_scope('inputs'):
                self._train_inputs = tf.placeholder(tf.int32,
                                                    shape=[self._batch_size])
                self._train_labels = tf.placeholder(tf.int32,
                                                    shape=[self._batch_size, 1])
            with tf.name_scope('embeddings'):
                self._embeddings = tf.Variable(
                    tf.random_uniform(shape=[self.vocab_size,
                                             self._embedding_size],
                                      minval=-1.0, maxval=1.0))
                embed = tf.nn.embedding_lookup(self._embeddings,
                                               self._train_inputs)
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
            tf.summary.scalar('loss', self._loss)
            with tf.name_scope('optimizer'):
                self._optimizer = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self._loss)
            self._merged = tf.summary.merge_all()
            self._tf_init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()
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
        self._id2word[0] = 'UNK'
        for word, freq in word_freq_dict.items():
            if freq >= self._min_count and word not in self._word2id:
                word_index += 1
                self._word2id[word] = word_index
                self._id2word[word_index] = word
        logger.info('Vocabulary size = {}'.format(len(self._word2id)))
        logger.info('Saving vocabulary to file {}'.format(output_filepath))
        with open(output_filepath, 'w', encoding='utf-8') as output_stream:
            for idx, word in self._id2word.items():
                print('{}#{}'.format(idx, word), file=output_stream)

    def load_vocab(self, vocab_filepath):
        """Load a previously saved vocabulary file."""
        with open(vocab_filepath, 'r') as vocab_stream:
            for line in vocab_stream:
                (idx, word) = line.strip().split('#', 1)
                self._word2id[word] = int(idx)
                self._id2word[int(idx)] = word

    def _get_lines(self, training_data_filepath):
        lines = []
        self._timers['lines_reading'] = time.monotonic()
        with open(training_data_filepath, 'r') as training_data_stream:
            for line in training_data_stream:
                lines.append(line.strip())
                if len(lines) == 100000:
                    logger.info('Lines reading time: {}s'.format(time.monotonic() - self._timers['lines_reading']))
                    self._timers['lines_reading'] = time.monotonic()
                    yield lines
                    lines = []
        logger.info('Lines reading time: {}s'.format(time.monotonic() - self._timers['lines_reading']))
        self._timers['lines_reading'] = time.monotonic()
        yield lines


    def _get_batches(self, training_data_filepath):
        """Return a generator over training batches."""
        batch = np.ndarray(shape=(self._batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self._batch_size, 1), dtype=np.int32)
        idx = 0
        # with open(training_data_filepath, 'r') as training_data_stream:
        #     for line in training_data_stream:
        for lines in self._get_lines(training_data_filepath):
            self._timers['batches_generation'] = time.monotonic()
            for line in lines:
                for target_id, target in enumerate(line.strip().split()):
                    for ctx_id, ctx in enumerate(line.strip().split()):
                        if ctx_id == target_id or abs(ctx_id - target_id) > self._window_size:
                            continue
                        batch[idx] = self._word2id[target]
                        labels[idx, 0] = self._word2id[ctx]
                        idx += 1
                        if idx == self._batch_size:
                            self._timings['batches_generation'] += \
                                time.monotonic() - self._timers['batches_generation']
                            self._timers['batches_generation'] = time.monotonic()
                            yield batch, labels
                            idx = 0
                            batch = np.ndarray(shape=(self._batch_size),
                                               dtype=np.int32)
                            labels = np.ndarray(shape=(self._batch_size, 1),
                                                dtype=np.int32)

    def train(self, training_data_filepath, model_dirpath):
        """Train over the data."""
        logger.info('Starting training...')

        sess_config = tf.ConfigProto()
        sess_config.intra_op_parallelism_threads = self._num_threads
        sess_config.inter_op_parallelism_threads = self._num_threads
        batch_count = 0
        with tf.Session(graph=self._graph, config=sess_config) as session:
            self._tf_init.run(session=session)  # Initialize all TF variables
            average_loss = 0
            for epoch in range(1, self._num_epochs + 1):
                step = 0
                for batch_inputs, batch_labels in self._get_batches(training_data_filepath):
                    step += 1
                    if epoch == 1:
                        batch_count += 1

                    self._timers['training'] = time.monotonic()
                    feed_dict = {self._train_inputs: batch_inputs,
                                 self._train_labels: batch_labels}
                    _, summary, loss_val = session.run(
                        [self._optimizer, self._merged, self._loss],
                        feed_dict=feed_dict)
                    average_loss += loss_val
                    self._timings['training'] += time.monotonic() - self._timers['training']

                    if step % 1000 == 0:
                        average_loss /= 1000
                        if epoch == 1:
                            logger.info('Epoch {}/{} average loss = {}'
                                        .format(epoch, self._num_epochs,
                                                average_loss))
                        else:
                            progress = (step / batch_count) * 100
                            logger.info('Epoch {}/{} progress = {}% average loss = {}'
                                        .format(epoch, self._num_epochs,
                                                progress, average_loss))
                        logger.info('Batches generation time: {}s'.format(self._timings['batches_generation']))
                        logger.info('Training time: {}s'.format(self._timings['training']))
                        self._timings['batches_generation'] = 0.
                        self._timings['training'] = 0.
                        average_loss = 0
            logger.info('Completed training. Saving model to {}'
                        .format(os.path.join(model_dirpath, 'model')))
            with tf.summary.FileWriter(model_dirpath, session.graph) as writer:
                self._saver.save(session, os.path.join(model_dirpath, 'model'))
