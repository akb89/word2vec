"""A word2vec implementation from scratch using Tensorflow."""

import os

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
                 num_threads, shuffling_buffer_size=100,
                 prefetch_batch_size=100, buffer_size=100):
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
        self._shuffling_buffer_size = shuffling_buffer_size
        self._prefetch_batch_size = prefetch_batch_size
        self._buffer_size = buffer_size

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

    def build_vocab(self, data_filepath, vocab_filepath):
        """Create vocabulary-related data."""
        logger.info('Building vocabulary from file {}'.format(data_filepath))
        word_freq_dict = defaultdict(int)
        with open(data_filepath, 'r') as data_stream:
            for line in data_stream:
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
        logger.info('Saving vocabulary to file {}'.format(vocab_filepath))
        with open(vocab_filepath, 'w', encoding='utf-8') as vocab_stream:
            for idx, word in self._id2word.items():
                print('{}#{}'.format(idx, word), file=vocab_stream)

    def load_vocab(self, vocab_filepath):
        """Load a previously saved vocabulary file."""
        with open(vocab_filepath, 'r') as vocab_stream:
            for line in vocab_stream:
                (idx, word) = line.strip().split('#', 1)
                self._word2id[word] = int(idx)
                self._id2word[int(idx)] = word

    def _get_dataset(self, training_data_filepath):
        def extract_skipgram_ex(line):
            def process_line(line):
                features = []
                labels = []
                tokens = line.strip().split()
                for target_id, target in enumerate(tokens):
                    for ctx_id, ctx in enumerate(tokens):
                        if ctx_id == target_id or abs(ctx_id - target_id) > self._window_size:
                            continue
                        features.append(self._word2id[target.decode('utf8')])
                        labels.append(self._word2id[ctx.decode('utf8')])
                return np.array([features, labels], dtype=np.int32)
                # return tf.stack([tf.convert_to_tensor(features, tf.int32),
                #                  tf.convert_to_tensor(labels, tf.int32)])
            return tf.py_func(process_line, [line], tf.int32)
        return (tf.data.TextLineDataset(training_data_filepath)
                .map(extract_skipgram_ex, num_parallel_calls=self._num_threads)
                .prefetch(self._buffer_size)
                .flat_map(lambda x: tf.data.Dataset.from_tensor_slices((x[0], x[1])))
                .shuffle(buffer_size=self._shuffling_buffer_size,
                         reshuffle_each_iteration=False)
                .repeat(self._num_epochs)
                .batch(self._batch_size)
                .prefetch(self._prefetch_batch_size))


    def train(self, training_data_filepath, model_dirpath):
        """Train over the data."""
        logger.info('Starting training...')

        sess_config = tf.ConfigProto()
        sess_config.intra_op_parallelism_threads = self._num_threads
        sess_config.inter_op_parallelism_threads = self._num_threads

        batch_count = 0
        with tf.Session(graph=self._graph, config=sess_config) as session:
            dataset = self._get_dataset(training_data_filepath)
            batches_iterator = dataset.make_initializable_iterator()
            init_op = batches_iterator.initializer
            batch_inputs, batch_labels = batches_iterator.get_next()
            #batch_labels = tf.reshape(batch_labels, [tf.size(batch_labels),1])
            batch_labels = tf.reshape(batch_labels, [-1, 1])
            self._tf_init.run(session=session)  # Initialize all TF variables

            average_loss = 0
            for epoch in range(1, self._num_epochs + 1):
                step = 0
                session.run(init_op)
                while True:
                    try:
                        step += 1
                        if epoch == 1:
                            batch_count += 1
                        feed_dict = {self._train_inputs: batch_inputs.eval(),
                                     self._train_labels: batch_labels.eval()}
                        _, summary, loss_val = session.run(
                            [self._optimizer, self._merged, self._loss],
                            feed_dict=feed_dict)
                        average_loss += loss_val
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
                            average_loss = 0
                    except tf.errors.OutOfRangeError:
                        break #End of epoch
            logger.info('Completed training. Saving model to {}'
                        .format(os.path.join(model_dirpath, 'model')))
            with tf.summary.FileWriter(model_dirpath, session.graph) as writer:
                self._saver.save(session, os.path.join(model_dirpath, 'model'))
