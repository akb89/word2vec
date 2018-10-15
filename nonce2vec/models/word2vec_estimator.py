"""A word2vec implementation from scratch using Tensorflow."""

import os

import logging

import math
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ('Word2Vec')


def cbow(features, labels, mode, params):
    """Return Word2Vec CBOW model."""
    pass


def skipgram(features, labels, mode, params):
    """Return Word2Vec Skipgram model."""
    # train_inputs = tf.placeholder(tf.int32, shape=[params['batch_size']])
    # train_labels = tf.placeholder(tf.int32, shape=[params['batch_size'], 1])
    # Should I reshape the labels?
    embeddings = tf.Variable(tf.random_uniform(
        shape=[params['vocab_size'], params['embedding_size']], minval=-1.0,
        maxval=1.0))
    input_embed = tf.nn.embedding_lookup(embeddings, features)
    nce_weights = tf.Variable(tf.truncated_normal(
        [params['vocab_size'], params['embedding_size']],
        stddev=1.0 / math.sqrt(params['embedding_size'])))
    nce_biases = tf.Variable(tf.zeros([params['vocab_size']]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=labels,
                                         inputs=input_embed,
                                         num_sampled=params['num_neg_samples'],
                                         num_classes=params['vocab_size']))
    optimizer = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)


class Word2Vec():
    """Tensorflow implementation of Word2vec."""

    def __init__(self, train_mode, min_count, batch_size, embedding_size,
                 num_neg_samples, learning_rate, window_size, num_epochs,
                 subsampling_rate, num_threads, vocab_filepath,
                 model_dirpath, shuffling_buffer_size=100,
                 prefetch_batch_size=10, buffer_size=10000,
                 data_filepath=None):
        if not os.path.exists(vocab_filepath):
            if not data_filepath:
                raise Exception(
                    'Unspecified data_filepath. You need to specify the data '
                    'file from which to build the vocabulary, or to specify a '
                    'valid vocabulary filepath')
            self._build_vocab(data_filepath, vocab_filepath)
        else:
            self._load_vocab(vocab_filepath)
        if train_mode != 'cbow' or train_mode != 'skipgram':
            raise Exception('Unsupported train_mode \'{}\''.format(train_mode))
        if train_mode == 'cbow':
            pass
        if train_mode == 'skipgram':
            self._estimator = tf.estimator.Estimator(
                model_fn=skipgram,
                model_dir=model_dirpath,
                params={
                    'vocab_size': len(self._word2id),
                    'embedding_size': embedding_size,
                    'num_neg_samples': num_neg_samples,
                    'learning_rate': learning_rate,
                })

    def _generate_dataset(self, training_data_filepath):
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

    def train(self, training_data_filepath):
        """Train Word2Vec."""
        self._estimator.train(
            input_fn=self._generate_dataset(training_data_filepath))

    def predict(self):
        """Predict."""
        pass
