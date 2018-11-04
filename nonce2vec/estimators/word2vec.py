"""A word2vec implementation using Tensorflow and estimators."""

import os

from collections import OrderedDict
import logging
import tensorflow as tf

import nonce2vec.utils.datasets as datasets_utils
import nonce2vec.models.word2vec as w2v_model

from nonce2vec.evaluation.men import MEN

from tensorflow.python import debug as tf_debug

logger = logging.getLogger(__name__)

__all__ = ('Word2Vec')


class Word2Vec():
    """Tensorflow implementation of Word2vec."""

    def __init__(self):
        """Initialize vocab dictionaries."""
        self._words = []
        self._counts = []
        self._total_count = 0

    @property
    def vocab_size(self):
        """Return the number of items in vocabulary.

        Since we use len(word_freq_dict) as the default index for UKN in
        the index_table, we have to add 1 to the length
        """
        return len(self._words) + 1

    def build_vocab(self, data_filepath, vocab_filepath, min_count):
        """Create vocabulary-related data."""
        logger.info('Building vocabulary from file {}'.format(data_filepath))
        logger.info('Loading word counts...')
        if self.vocab_size > 1:
            logger.warning('This instance of W2V\'s vocabulary does not seem '
                           'to be empty. Erasing previously stored vocab...')
            self._words, self._counts, self._total_count = [], [], 0
        word_count_dict = OrderedDict()
        with open(data_filepath, 'r') as data_stream:
            for line in data_stream:
                for word in line.strip().split():
                    if word not in word_count_dict:
                        word_count_dict[word] = 1
                    else:
                        word_count_dict[word] += 1
        logger.info('Saving word frequencies to file: {}'.format(vocab_filepath))
        with open(vocab_filepath, 'w') as vocab_stream:
            for word, count in word_count_dict.items():
                print('{}\t{}'.format(word, count), file=vocab_stream)
                if count >= min_count:
                    self._words.append(word)
                    self._counts.append(count)
                    self._total_count += count

    def load_vocab(self, vocab_filepath, min_count):
        """Load a previously saved vocabulary file."""
        logger.info('Loading word counts from file {}'.format(vocab_filepath))
        self._words, self._counts, self._total_count = [], [], 0
        with open(vocab_filepath, 'r', encoding='UTF-8') as vocab_stream:
            for line in vocab_stream:
                word_count = line.strip().split('\t', 1)
                word, count = word_count[0], int(word_count[1])
                if count >= min_count:
                    self._words.append(word)
                    self._counts.append(count)
                    self._total_count += count
        logger.info('Done loading word counts')

    def train(self, train_mode, training_data_filepath, model_dirpath,
              batch_size, embedding_size, num_neg_samples,
              learning_rate, window_size, num_epochs, sampling_rate,
              p_num_threads, t_num_threads, shuffling_buffer_size,
              save_summary_steps, save_checkpoints_steps, keep_checkpoint_max,
              log_step_count_steps, debug=False, debug_hook_name=None,
              xla=False):
        """Train Word2Vec."""
        if self.vocab_size == 1:
            raise Exception('You need to build or load a vocabulary before '
                            'training word2vec')
        if train_mode not in ('cbow', 'skipgram'):
            raise Exception('Unsupported train_mode \'{}\''.format(train_mode))
        sess_config = tf.ConfigProto(log_device_placement=True)
        sess_config.intra_op_parallelism_threads = t_num_threads
        sess_config.inter_op_parallelism_threads = t_num_threads
        # if xla:
        #     sess_config.graph_options.optimizer_options.global_jit_level = \
        #      tf.OptimizerOptions.ON_1  # JIT compilation on GPU
        run_config = tf.estimator.RunConfig(
            session_config=sess_config, save_summary_steps=save_summary_steps,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=keep_checkpoint_max,
            log_step_count_steps=log_step_count_steps)
        estimator = tf.estimator.Estimator(
            model_fn=w2v_model.model,
            model_dir=model_dirpath,
            config=run_config,
            params={
                'mode': train_mode,
                'vocab_size': self.vocab_size,
                'batch_size': batch_size,
                'embedding_size': embedding_size,
                'num_neg_samples': num_neg_samples,
                'learning_rate': learning_rate,
                'words': self._words,
                'p_num_threads': p_num_threads,
                'xla': xla,
                'men': MEN(os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'resources', 'MEN_dataset_natural_form_full'))
            })
        if debug:
            if not debug_hook_name:
                raise Exception('You need to specify the --debug-hook-name '
                                'parameter when passing --debug')
            hooks = [tf.train.ProfilerHook(
                save_steps=save_summary_steps, show_dataflow=True,
                show_memory=True, output_dir=model_dirpath),
                     tf_debug.TensorBoardDebugHook(debug_hook_name)]
        else:
            hooks = [tf.train.ProfilerHook(
                save_steps=save_summary_steps, show_dataflow=True,
                show_memory=True, output_dir=model_dirpath)]
        estimator.train(
            input_fn=lambda: datasets_utils.get_w2v_train_dataset(
                training_data_filepath, train_mode, self._words, self._counts,
                self._total_count, window_size, sampling_rate, batch_size,
                num_epochs, p_num_threads, shuffling_buffer_size),
            hooks=hooks)