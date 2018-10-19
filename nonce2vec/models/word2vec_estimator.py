"""A word2vec implementation using Tensorflow and estimators."""

import os

from collections import defaultdict

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
    labels = tf.reshape(labels, [-1, 1])
    with tf.name_scope('embeddings'):
        embeddings = tf.Variable(tf.random_uniform(
            shape=[params['vocab_size'], params['embedding_size']],
            minval=-1.0, maxval=1.0))
        input_embed = tf.nn.embedding_lookup(embeddings, features)
    with tf.name_scope('weights'):
        nce_weights = tf.Variable(tf.truncated_normal(
            [params['vocab_size'], params['embedding_size']],
            stddev=1.0 / math.sqrt(params['embedding_size'])))
    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([params['vocab_size']]))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=labels,
                           inputs=input_embed,
                           num_sampled=params['num_neg_samples'],
                           num_classes=params['vocab_size']))
    with tf.name_scope('optimizer'):
        optimizer = (tf.train.GradientDescentOptimizer(params['learning_rate'])
                     .minimize(loss, global_step=tf.train.get_global_step()))
    men_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'resources', 'MEN_dataset_natural_form_full')
    with open(men_filepath, 'r') as men_stream:
        first_men_labels = []
        second_men_labels = []
        men_sim_labels = []
        for line in men_stream:
            tokens = line.strip().split()
            first_men_labels.append(params['word2id'][tokens[0]])
            second_men_labels.append(params['word2id'][tokens[1]])
            men_sim_labels.append(float(tokens[2]))
        first_men_labels = tf.convert_to_tensor(first_men_labels)
        second_men_labels = tf.convert_to_tensor(second_men_labels)
        men_sim_labels = tf.convert_to_tensor(men_sim_labels)
    first_men_embeddings = tf.nn.embedding_lookup(embeddings, first_men_labels)
    second_men_embeddings = tf.nn.embedding_lookup(embeddings, second_men_labels)
    men_sim_predictions = tf.losses.cosine_distance(first_men_embeddings, second_men_embeddings, axis=1, reduction=tf.losses.Reduction.NONE)
    men_correlation = tf.contrib.metrics.streaming_pearson_correlation(men_sim_predictions, men_sim_labels)
    metrics = {'MEN': men_correlation}
    tf.summary.scalar('MEN', men_correlation[1])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer,
                                      eval_metric_ops=metrics)
    #return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)


class Word2Vec():
    """Tensorflow implementation of Word2vec."""

    def __init__(self):
        """Initialize vocab dictionaries."""
        self._id2word = {}
        self._word2id = defaultdict(lambda: 0)

    @property
    def vocab_size(self):
        """Return vocabulary length."""
        return len(self._word2id)

    def build_vocab(self, data_filepath, vocab_filepath, min_count):
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
            if freq >= min_count and word not in self._word2id:
                word_index += 1
                self._word2id[word] = word_index
                self._id2word[word_index] = word
        logger.info('Vocabulary size = {}'.format(self.vocab_size))
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


    def _generate_train_dataset(self, training_data_filepath, window_size,
                                batch_size, num_epochs,
                                shuffling_buffer_size=1000,
                                buffer_size=1000):
        def get_ctx_items(tokens, target_id, window_size):
            ctx_1 = tokens[max(0, target_id-window_size):target_id]
            ctx_2 = tokens[target_id+1:min(len(tokens), target_id+window_size+1)]
            return [*ctx_1, *ctx_2]

        def get_example():
            with open(training_data_filepath, 'r') as training_datastream:
                for line in training_datastream:
                    tokens = line.strip().split()
                    for target_id, target in enumerate(tokens):
                        for ctx in get_ctx_items(tokens, target_id, window_size):
                            yield self._word2id[target], self._word2id[ctx]

        return (tf.data.Dataset.from_generator(get_example,
                                               (tf.int32, tf.int32))
                .shuffle(buffer_size=shuffling_buffer_size,
                         reshuffle_each_iteration=False)
                .repeat(num_epochs)
                .batch(batch_size)
                .prefetch(buffer_size))

    def train(self, train_mode, training_data_filepath, model_dirpath,
              batch_size, embedding_size, num_neg_samples, learning_rate,
              window_size, num_epochs, subsampling_rate, num_threads):
        """Train Word2Vec."""
        if self.vocab_size == 0:
            raise Exception('You need to build or load a vocabulary before '
                            'training word2vec')
        if train_mode not in ('cbow', 'skipgram'):
            raise Exception('Unsupported train_mode \'{}\''.format(train_mode))
        if train_mode == 'cbow':
            pass
        if train_mode == 'skipgram':
            sess_config = tf.ConfigProto()
            sess_config.intra_op_parallelism_threads = num_threads
            sess_config.inter_op_parallelism_threads = num_threads
            run_config = tf.estimator.RunConfig(session_config=sess_config)
            self._estimator = tf.estimator.Estimator(
                model_fn=skipgram,
                model_dir=model_dirpath,
                config=run_config,
                params={
                    'vocab_size': self.vocab_size,
                    'embedding_size': embedding_size,
                    'num_neg_samples': num_neg_samples,
                    'learning_rate': learning_rate,
                    'word2id': self._word2id
                })
        self._estimator.train(
            input_fn=lambda: self._generate_train_dataset(
                training_data_filepath, window_size, batch_size, num_epochs))

    def _generate_eval_dataset(self):
        men_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'resources',
                                    'MEN_dataset_natural_form_full')
        with open(men_filepath, 'r') as men_stream:
            pass


    def evaluate(self):
        """Evaluate Word2Vec against the MEN dataset."""
        eval_result = self._estimator.evaluate(
            input_fn=self._generate_eval_dataset)
        logger.info('MEN correlation ratio: {}'.format(**eval_result))
