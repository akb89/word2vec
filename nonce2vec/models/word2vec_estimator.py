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
    pass


class Word2Vec():
    """Tensorflow implementation of Word2vec."""

    def __init__(self, train_mode, model_dirpath):
        if train_mode != 'cbow' or train_mode != 'skipgram':
            raise Exception('Unsupported train_mode \'{}\''.format(train_mode))
        if train_mode == 'cbow':
            pass
        if train_mode == 'skipgram':
            self._estimator = tf.estimator.Estimator(
                model_fn=skipgram,
                params={
                    'feature_columns': my_feature_columns,
                    # Two hidden layers of 10 nodes each.
                    'hidden_units': [10, 10],
                    # The model must choose between 3 classes.
                    'n_classes': 3,
                })

    def _get_dataset(training_data_filepath):
        pass

    def train(self, training_data_filepath):
        self._estimator.train(input_fn=self._get_dataset(training_data_filepath))
