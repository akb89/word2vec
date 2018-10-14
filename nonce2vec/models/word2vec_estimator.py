"""A word2vec implementation from scratch using Tensorflow."""

import os

from collections import defaultdict

import logging

import math
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ('Word2Vec')


def cbow(features, labels, mode, params):
    """Return Word2Vec CBOW model."""
    pass


def skipgram(features, labels, mode, params):
    """Return Word2Vec Skipgram model."""
    # train_inputs = tf.placeholder(tf.int32, shape=[params['batch_size']])
    # train_labels = tf.placeholder(tf.int32, shape=[params['batch_size'], 1])
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

    def __init__(self, train_mode, model_dirpath, embedding_size,
                 num_neg_samples, learning_rate, vocab_filepath):
        if not vocab_filepath:
            # build vocab
            pass
        else:
            # load vocab
            pass
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

    def _get_dataset(self, training_data_filepath):
        dataset = tf.data.TextLineDataset(training_data_filepath).flat_map(self._string_to_skip_gram)
        if self._perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(self._prefetch_batches_size)
        return dataset

    def train(self, training_data_filepath):
        self._estimator.train(input_fn=self._get_dataset(training_data_filepath))
