"""Word2Vec cbow implementation."""

import logging

import math
import tensorflow as tf

import nonce2vec.utils.vocab as vocab_utils

logger = logging.getLogger(__name__)

__all__ = ('get_model', 'get_train_dataset')


def ctx_idxx(target_idx, window_size, tokens):
    ctx_range = tf.range(start=tf.maximum(tf.constant(0, dtype=tf.int32),
                                          target_idx-window_size),
                         limit=tf.minimum(tf.size(tokens, out_type=tf.int32),
                                          target_idx+window_size+1),
                         delta=1, dtype=tf.int32)
    idx = tf.case({tf.less_equal(target_idx, window_size): lambda: target_idx,
                   tf.greater(target_idx, window_size): lambda: window_size},
                  exclusive=True)
    t0 = lambda: tf.constant([], dtype=tf.int32)
    t1 = lambda: ctx_range[idx+1:]
    t2 = lambda: ctx_range[0:idx]
    t3 = lambda: tf.concat([ctx_range[0:idx], ctx_range[idx+1:]], axis=0)
    c1 = tf.logical_and(tf.equal(idx, 0),
                        tf.less(idx+1, tf.size(ctx_range, out_type=tf.int32)))
    c2 = tf.logical_and(tf.greater(idx, 0),
                        tf.equal(idx+1, tf.size(ctx_range, out_type=tf.int32)))
    c3 = tf.logical_and(tf.greater(idx, 0),
                        tf.less(idx+1, tf.size(ctx_range, out_type=tf.int32)))
    return tf.case({c1: t1, c2: t2, c3: t3}, default=t0, exclusive=True)


def stack_to_features_and_labels(features, labels, target_idx, tokens,
                                 window_size):
    ctxs = ctx_idxx(target_idx, window_size, tokens)
    ctx_features = tf.gather(tokens, ctxs)
    diff = 2 * window_size - tf.size(ctx_features)
    ctx_features = tf.reshape(ctx_features, [1, -1])
    paddings = tf.concat([tf.constant([[0, 0]]), tf.concat([tf.constant([[0]]), [[diff]]], axis=1)], axis=0)
    padded_ctx_features = tf.pad(ctx_features, paddings,
                                 constant_values='_CBOW#_!MASK_')
    label = tokens[target_idx]
    return tf.concat([features, padded_ctx_features], axis=0), \
           tf.concat([labels, [label]], axis=0), target_idx+1, tokens, \
           window_size


def extract_examples(tokens, window_size, p_num_threads):
    features = tf.constant([], shape=[0, 2*window_size], dtype=tf.string)
    labels = tf.constant([], dtype=tf.string)
    target_idx = tf.constant(0, dtype=tf.int32)
    t_window_size = tf.constant(window_size, dtype=tf.int32)
    max_size = tf.size(tokens, out_type=tf.int32)  # TODO: check if not -1
    target_idx_less_than_tokens_size = lambda w, x, y, z, k: tf.less(y, max_size)
    result = tf.while_loop(
        cond=target_idx_less_than_tokens_size,
        body=stack_to_features_and_labels,
        loop_vars=[features, labels, target_idx, tokens, t_window_size],
        shape_invariants=[tf.TensorShape([None, 2*window_size]),
                          tf.TensorShape([None]),
                          target_idx.get_shape(), tokens.get_shape(),
                          t_window_size.get_shape()],
        parallel_iterations=p_num_threads)
    return result[0], result[1]


def sample_prob(tokens, sampling_rate, word_freq_table):
    """Sample according to w2v paper formula: p = 1 - sqrt(t/f)."""
    return 1 - tf.sqrt(tf.fill([tf.size(tokens)], sampling_rate) / word_freq_table.lookup(tokens))


def filter_tokens(tokens, min_count, sampling_rate, word_count_table,
                  word_freq_table):
    return tf.logical_and(
        tf.greater_equal(word_count_table.lookup(tokens),
                         tf.fill([tf.size(tokens)],
                                 tf.constant(min_count))),
        tf.less(sample_prob(tokens, sampling_rate, word_freq_table),
                tf.random_uniform(shape=[tf.size(tokens)],
                                  minval=0, maxval=1, dtype=tf.float32)))


def get_train_dataset(training_data_filepath, window_size, batch_size,
                      num_epochs, p_num_threads, shuffling_buffer_size):
    """Generate a Tensorflow Dataset for a CBOW model."""
    return (tf.data.TextLineDataset(training_data_filepath)
            .map(tf.strings.strip, num_parallel_calls=p_num_threads)
            #.filter(lambda x: tf.not_equal(tf.strings.length(x), 0))  # Filter empty strings
            .map(lambda x: tf.strings.split([x]),
                 num_parallel_calls=p_num_threads)
            .filter(lambda x: filter_tokens(x.values, min_count, sampling_rate,
                                            word_count_table, word_freq_table))
            .filter(lambda x: tf.greater(tf.size(x), 1))
            .map(lambda x: extract_examples(x.values, window_size,
                                            p_num_threads),
                 num_parallel_calls=p_num_threads)
            .flat_map(lambda features, labels: tf.data.Dataset.from_tensor_slices((features, labels)))
            .shuffle(buffer_size=shuffling_buffer_size,
                     reshuffle_each_iteration=False)
            .repeat(num_epochs)
            .batch(batch_size, drop_remainder=True))
            # we need drop_remainder to statically know the batch dimension
            # this is required to get features.get_shape()[0] in avg_ctx_features


def stack_mean_to_avg_tensor(features, vocab, embeddings):
    def internal_func(avg, idx):
        # For a given set of features, corresponding to a given set
        # of context words
        feat_row = features[idx]
        # select only valid context words
        is_valid_string = tf.not_equal(feat_row, '_CBOW#_!MASK_')
        valid_feats = tf.boolean_mask(feat_row, is_valid_string)
        # discretized the features
        discretized_feats = vocab.lookup(valid_feats)
        # select their corresponding embeddings
        embedded_feats = tf.nn.embedding_lookup(embeddings, discretized_feats)
        # average over the given context word embeddings
        mean = tf.reduce_mean(embedded_feats, 0)
        # concatenate to the return averaged tensor stacking all
        # averaged context embeddings for a given batch
        avg = tf.concat([avg, [mean]], axis=0)
        return [avg, idx+1]
    return internal_func


def avg_ctx_features(features, embeddings, vocab, p_num_threads):
    feat_batch_size = features.get_shape()[0]
    embedding_size = embeddings.get_shape()[1]
    idx_within_batch_size = lambda x, y: tf.less(y, feat_batch_size)
    stack_mean_to_avg_tensor_with_vocab = stack_mean_to_avg_tensor(features, vocab, embeddings)
    avg = tf.constant([], shape=[0, embedding_size], dtype=tf.float32)
    idx = tf.constant(0, dtype=tf.int32)
    avg, idx = tf.while_loop(
        cond=idx_within_batch_size,
        body=stack_mean_to_avg_tensor_with_vocab,
        loop_vars=[avg, idx],
        shape_invariants=[tf.TensorShape([None, embedding_size]),
                          idx.get_shape()],
        parallel_iterations=p_num_threads,
        maximum_iterations=feat_batch_size)
    return avg


def get_model(features, labels, mode, params):
    """Return Word2Vec cbow model for a Tensorflow Estimator."""
    with tf.contrib.compiler.jit.experimental_jit_scope():
        vocab = vocab_utils.get_tf_vocab_table(params['word_freq_dict'])
        word_count = vocab_utils.get_tf_word_count_table(params['word_freq_dict'])
        with tf.name_scope('hidden'):
            embeddings = tf.get_variable(
                'embeddings', shape=[params['vocab_size'],
                                     params['embedding_size']],
                initializer=tf.random_uniform_initializer(minval=-1.0,
                                                          maxval=1.0))

        discretized_labels = tf.reshape(vocab.lookup(labels), [-1, 1])
        discretized_avg_features = avg_ctx_features(
            features, embeddings, vocab, params['p_num_threads'])

        with tf.name_scope('weights'):
            nce_weights = tf.get_variable(
                'nce_weights', shape=[params['vocab_size'],
                                      params['embedding_size']],
                initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / math.sqrt(params['embedding_size'])))

        with tf.name_scope('biases'):
            nce_biases = tf.get_variable(
                'nce_biases', shape=[params['vocab_size']],
                initializer=tf.zeros_initializer)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=discretized_labels,
                               inputs=discretized_avg_features,
                               num_sampled=params['num_neg_samples'],
                               num_classes=params['vocab_size']))

        with tf.name_scope('optimizer'):
            optimizer = (tf.train.GradientDescentOptimizer(
                params['learning_rate'])
                         .minimize(loss,
                                   global_step=tf.train.get_global_step()))

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)
