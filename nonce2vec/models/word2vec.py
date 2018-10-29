"""A word2vec implementation using Tensorflow and estimators."""

import os

from collections import OrderedDict
import logging
import tensorflow as tf

import nonce2vec.learning.skipgram as skipgram

from nonce2vec.evaluation.men import MEN

logger = logging.getLogger(__name__)

__all__ = ('Word2Vec')


class Word2Vec():
    """Tensorflow implementation of Word2vec."""

    def __init__(self):
        """Initialize vocab dictionaries."""
        # Needs to make sure dict is ordered to always get the same
        # lookup table
        self._word_freq_dict = OrderedDict()

    @property
    def vocab_size(self):
        """Return the number of items in vocabulary.

        Since we use len(word_freq_dict) as the default index for UKN in
        the index_table, we have to add 1 to the length
        """
        return len(self._word_freq_dict) + 1

    def build_vocab(self, data_filepath, vocab_filepath):
        """Create vocabulary-related data."""
        logger.info('Building vocabulary from file {}'.format(data_filepath))
        logger.info('Loading word frequencies...')
        if self.vocab_size > 1:
            logger.warning('This instance of W2V\'s vocabulary does not seem '
                           'to be empty. Erasing previously stored vocab...')
        with open(data_filepath, 'r') as data_stream:
            for line in data_stream:
                for word in line.strip().split():
                    if word not in self._word_freq_dict:
                        self._word_freq_dict[word] = 1
                    else:
                        self._word_freq_dict[word] += 1
        logger.info('Saving word frequencies to file: {}'.format(vocab_filepath))
        with open(vocab_filepath, 'w') as vocab_stream:
            for key, value in self._word_freq_dict.items():
                print('{}\t{}'.format(key, value), file=vocab_stream)

    def load_vocab(self, vocab_filepath):
        """Load a previously saved vocabulary file."""
        logger.info('Loading word frequencies from file {}'
                    .format(vocab_filepath))
        with open(vocab_filepath, 'r', encoding='UTF-8') as vocab_stream:
            for line in vocab_stream:
                word_freq = line.strip().split('\t', 1)
                self._word_freq_dict[word_freq[0]] = int(word_freq[1])

    def train(self, train_mode, training_data_filepath, model_dirpath,
              min_count, batch_size, embedding_size, num_neg_samples,
              learning_rate, window_size, num_epochs, subsampling_rate,
              p_num_threads, t_num_threads, save_summary_steps,
              save_checkpoints_steps, keep_checkpoint_max,
              log_step_count_steps):
        """Train Word2Vec."""
        if self.vocab_size == 1:
            raise Exception('You need to build or load a vocabulary before '
                            'training word2vec')
        if train_mode not in ('cbow', 'skipgram'):
            raise Exception('Unsupported train_mode \'{}\''.format(train_mode))
        sess_config = tf.ConfigProto(log_device_placement=True)
        sess_config.intra_op_parallelism_threads = t_num_threads
        sess_config.inter_op_parallelism_threads = t_num_threads
        sess_config.graph_options.optimizer_options.global_jit_level = \
         tf.OptimizerOptions.ON_1  # JIT compilation on GPU
        run_config = tf.estimator.RunConfig(
            session_config=sess_config, save_summary_steps=save_summary_steps,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=keep_checkpoint_max,
            log_step_count_steps=log_step_count_steps)
        if train_mode == 'cbow':
            pass
        if train_mode == 'skipgram':
            estimator = tf.estimator.Estimator(
                model_fn=skipgram.get_model,
                model_dir=model_dirpath,
                config=run_config,
                params={
                    'vocab_size': self.vocab_size,
                    'embedding_size': embedding_size,
                    'num_neg_samples': num_neg_samples,
                    'learning_rate': learning_rate,
                    'word_freq_dict': self._word_freq_dict,
                    'min_count': min_count,
                    'men': MEN(os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        'resources', 'MEN_dataset_natural_form_full'))
                })
            estimator.train(
                input_fn=lambda: skipgram.get_train_dataset(
                    training_data_filepath, window_size, batch_size,
                    num_epochs, p_num_threads), hooks=[tf.train.ProfilerHook(
                        save_steps=100, show_dataflow=True, show_memory=True,
                        output_dir=model_dirpath)])
