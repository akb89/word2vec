"""A word2vec implementation using Tensorflow and estimators."""

import os

from collections import defaultdict
import logging
import tensorflow as tf

# from tensorflow.python import debug as tf_debug  # pylint: disable=E0611

import word2vec.utils.datasets as datasets_utils
import word2vec.models.word2vec as w2v_model

from word2vec.evaluation.men import MEN

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
        word_count_dict = defaultdict(int)
        with open(data_filepath, 'r') as data_stream:
            for line in data_stream:
                for word in line.strip().split():
                    word_count_dict[word] += 1
        logger.info('Saving word frequencies to file: {}'
                    .format(vocab_filepath))
        with open(vocab_filepath, 'w') as vocab_stream:
            # words need to be sorted in decreasing frequency to be able
            # to rely on the default tf.nn.log_uniform_candidate_sampler
            # later on in the tf.nn.nce_loss
            for word, count in sorted(word_count_dict.items(),
                                      key=lambda x: x[1], reverse=True):
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

    # pylint: disable=R0914,W0613
    def train(self, train_mode, training_data_filepath, model_dirpath,
              batch_size, embedding_size, num_neg_samples,
              learning_rate, window_size, num_epochs, sampling_rate,
              p_num_threads, t_num_threads, shuffling_buffer_size,
              save_summary_steps, save_checkpoints_steps, keep_checkpoint_max,
              log_step_count_steps, debug, debug_port, xla):
        """Train Word2Vec."""
        if self.vocab_size == 1:
            raise Exception('You need to build or load a vocabulary before '
                            'training word2vec')
        if train_mode not in ('cbow', 'skipgram'):
            raise Exception('Unsupported train_mode \'{}\''.format(train_mode))
        sess_config = tf.compat.v1.ConfigProto(log_device_placement=True)
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
        # waiting for v2 fix in tf.summary.FileWriter:
        tf.compat.v1.disable_eager_execution()
        if debug:
            raise Exception('Unsupported parameter: waiting for the TF team '
                            'to release v2 equivalents for TensorBoardDebugHook')
            # hooks = [tf.estimator.ProfilerHook(
            #     save_steps=save_summary_steps, show_dataflow=True,
            #     show_memory=True, output_dir=model_dirpath),
            #          tf_debug.TensorBoardDebugHook('localhost:{}'
            #                                        .format(debug_port))]
        # else:
        hooks = [tf.estimator.ProfilerHook(
            save_steps=save_summary_steps, show_dataflow=True,
            show_memory=True, output_dir=model_dirpath)]
        estimator.train(
            input_fn=lambda: datasets_utils.get_w2v_train_dataset(
                training_data_filepath, train_mode, self._words, self._counts,
                self._total_count, window_size, sampling_rate, batch_size,
                num_epochs, p_num_threads, shuffling_buffer_size),
            hooks=hooks)
