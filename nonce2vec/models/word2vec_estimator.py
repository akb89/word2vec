"""A word2vec implementation using Tensorflow and estimators."""

import os

from collections import defaultdict

import logging

import math
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ('Word2Vec')

class TrainingHook(tf.train.SessionRunHook):
    pass


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
    # with open(men_filepath, 'r') as men_stream:
    #     first_men_labels = []
    #     second_men_labels = []
    #     men_sim_labels = []
    #     for line in men_stream:
    #         tokens = line.strip().split()
    #         first_men_labels.append(params['vocab'].lookup(tokens[0]))
    #         second_men_labels.append(params['vocab'].lookup(tokens[1]))
    #         men_sim_labels.append(float(tokens[2]))
    #     first_men_labels = tf.convert_to_tensor(first_men_labels)
    #     second_men_labels = tf.convert_to_tensor(second_men_labels)
    #     men_sim_labels = tf.convert_to_tensor(men_sim_labels)
    # first_men_embeddings = tf.nn.embedding_lookup(embeddings, first_men_labels)
    # second_men_embeddings = tf.nn.embedding_lookup(embeddings, second_men_labels)
    # men_sim_predictions = tf.losses.cosine_distance(first_men_embeddings, second_men_embeddings, axis=1, reduction=tf.losses.Reduction.NONE)
    # men_correlation = tf.contrib.metrics.streaming_pearson_correlation(men_sim_predictions, men_sim_labels)
    # metrics = {'MEN': men_correlation}
    # tf.summary.scalar('MEN', men_correlation[1])
    # return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer,
    #                                   eval_metric_ops=metrics)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)


class Word2Vec():
    """Tensorflow implementation of Word2vec."""

    def __init__(self):
        """Initialize vocab dictionaries."""
        self._vocab = None
        self._word_freq_dict = defaultdict(int)

    @property
    def vocab_size(self):
        return len(self._word_freq_dict)

    def _get_tf_vocab_table(self, word_freq_dict, min_count):
        mapping_strings = tf.constant([word for (word, freq) in word_freq_dict.items() if freq >= min_count])
        return tf.contrib.lookup.index_table_from_tensor(
            mapping=mapping_strings, num_oov_buckets=0, default_value=0)

    def build_vocab(self, data_filepath, vocab_filepath):
        """Create vocabulary-related data."""
        logger.info('Building vocabulary from file {}'.format(data_filepath))
        logger.info('Loading word frequencies...')
        if self.vocab_size != 0:
            logger.warning('This instance of W2V\'s vocabulary does not seem '
                           'to be empty. Erasing previously stored vocab...')
        with open(data_filepath, 'r') as data_stream:
            for line in data_stream:
                for word in line.strip().split():
                    self._word_freq_dict[word] += 1
        logger.info('Saving word frequencies to file: {}'.format(vocab_filepath))
        with open(vocab_filepath, 'w') as vocab_stream:
            for key, value in self._word_freq_dict.items():
                print('{}\t{}'.format(key, value), file=vocab_stream)

    def load_vocab(self, vocab_filepath):
        """Load a previously saved vocabulary file."""
        logger.info('Loading word frequencies from file {}'.format(vocab_filepath))
        with open(vocab_filepath, 'r', encoding='UTF-8') as vocab_stream:
            for line in vocab_stream:
                word_freq = line.strip().split('\t', 1)
                self._word_freq_dict[word_freq[0]] = int(word_freq[1])

    def _generate_train_dataset(self, training_data_filepath, window_size,
                                min_count, batch_size, num_epochs,
                                p_num_threads, shuffling_buffer_size=10,
                                prefetch_batch_size=10):
        # Needs to be here to make sure everything belongs to the same graph
        self._vocab = self._get_tf_vocab_table(self._word_freq_dict, min_count)
        def ctx_idxx(target_idx, window_size, tokens):
            """
            if target_idx > 0 and target_idx+1 < tf.size(ctx_range):
                return tf.concat([ctx_range[0:target_idx],
                                  ctx_range[target_idx+1, tf.size(ctx_range)]], axis=0)
            if target_idx == 0 and target_idx+1 < tf.size(ctx_range):
                return ctx_range[target_idx+1, tf.size(ctx_range)]
            if target_idx > 0 and target_idx+1 == tf.size(ctx_range):
                return ctx_range[0:target_idx]
            return tf.constant([])
            """
            ctx_range = tf.range(start=tf.maximum(tf.constant(0, dtype=tf.int64), target_idx-window_size),
                                 limit=tf.minimum(tf.size(tokens, out_type=tf.int64),
                                                  target_idx+window_size+1),
                                 delta=1, dtype=tf.int64)
            t0 = lambda: tf.constant([], dtype=tf.int64)
            t1 = lambda: tf.concat([ctx_range[0:target_idx],
                                    ctx_range[target_idx+1:tf.size(ctx_range)]],
                                   axis=0)
            t2 = lambda: ctx_range[target_idx+1:tf.size(ctx_range)]
            t3 = lambda: ctx_range[0:target_idx]
            c1 = tf.logical_and(tf.greater(target_idx, 0),
                                tf.less(target_idx+1, tf.size(ctx_range, out_type=tf.int64)))
            c2 = tf.logical_and(tf.equal(target_idx, 0),
                                tf.less(target_idx+1, tf.size(ctx_range, out_type=tf.int64)))
            c3 = tf.logical_and(tf.greater(target_idx, 0),
                                tf.equal(target_idx+1, tf.size(ctx_range, out_type=tf.int64)))
            return tf.case({c1: t1, c2: t2, c3: t3}, default=t0, exclusive=True)

        def stack_to_features_and_labels(features, labels, target_idx, tokens):
            ctxs = ctx_idxx(target_idx, 2, tokens)
            label = tf.nn.embedding_lookup(tokens, ctxs)
            feature = tf.fill([tf.size(label)], tokens[target_idx])
            return tf.concat([features, feature], axis=0), \
                   tf.concat([labels, label], axis=0), target_idx+1, tokens

        def extract_examples(tokens, window_size, p_num_threads):
            features = tf.constant([], dtype=tf.int64)
            labels = tf.constant([], dtype=tf.int64)
            target_idx = tf.constant(0, dtype=tf.int64)
            target_idx_less_than_tokens_size = lambda w, x, y, z: tf.less(y, tf.size(tokens, out_type=tf.int64))
            result = tf.while_loop(
                cond=target_idx_less_than_tokens_size,
                body=stack_to_features_and_labels,
                loop_vars=[features, labels, target_idx, tokens],
                shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None]),
                                  target_idx.get_shape(), tokens.get_shape()],
                parallel_iterations=p_num_threads)
            return result[0], result[1]

        return (tf.data.TextLineDataset(training_data_filepath)
                .map(tf.strings.strip, num_parallel_calls=p_num_threads)
                .filter(lambda x: tf.not_equal(tf.strings.length(x), 0))  # Filter empty strings
                .map(lambda x: tf.strings.split([x]), num_parallel_calls=p_num_threads)
                .map(lambda x: self._vocab.lookup(x.values), num_parallel_calls=p_num_threads)  # discretize
                .map(lambda tokens: extract_examples(tokens, window_size, p_num_threads), num_parallel_calls=p_num_threads)
                .prefetch(25000)
                .flat_map(lambda features, labels: tf.data.Dataset.from_tensor_slices((features, labels)))
                .shuffle(buffer_size=shuffling_buffer_size,
                         reshuffle_each_iteration=False)
                .repeat(num_epochs)
                .batch(batch_size)
                .prefetch(prefetch_batch_size))

    def train(self, train_mode, training_data_filepath, model_dirpath,
              min_count, batch_size, embedding_size, num_neg_samples,
              learning_rate, window_size, num_epochs, subsampling_rate,
              p_num_threads, t_num_threads):
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
            sess_config.intra_op_parallelism_threads = t_num_threads
            sess_config.inter_op_parallelism_threads = t_num_threads
            run_config = tf.estimator.RunConfig(session_config=sess_config)
            self._estimator = tf.estimator.Estimator(
                model_fn=skipgram,
                model_dir=model_dirpath,
                config=run_config,
                params={
                    'vocab_size': self.vocab_size,
                    'embedding_size': embedding_size,
                    'num_neg_samples': num_neg_samples,
                    'learning_rate': learning_rate
                    #'vocab': self._vocab
                })
        self._estimator.train(
            input_fn=lambda: self._generate_train_dataset(
                training_data_filepath, window_size, min_count, batch_size,
                num_epochs, p_num_threads))
        # tf.enable_eager_execution()
        # with tf.Session(graph=tf.Graph()) as session:
        #     dataset = self._generate_train_dataset(
        #             training_data_filepath, window_size, batch_size,
        #             num_epochs, p_num_threads)
        #     iterator = dataset.make_initializable_iterator()
        #     init_op = iterator.initializer
        #     x = iterator.get_next()
        #     session.run(init_op)
        #     while True:
        #         print(x[0].eval(), x[1].eval())

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
