"""Word2Vec skipgram-related implementations."""

import logging

import math
import tensorflow as tf

import nonce2vec.utils.vocab as vocab_utils

logger = logging.getLogger(__name__)

__all__ = ('get_model', 'get_train_dataset')


def ctx_idxx(target_idx, window_size, tokens):
    """
    # Get the idx corresponding to target-idx in the ctx_range:
    if target_idx - window_size <= 0:
        idx = target_idx
    if target_idx - window_size > 0:
        idx = window_size
    # We would like to return the ctx_range minus the idx, to remove the target:
    return ctx_range[0:idx] + ctx_range[idx+1:]
    # Let us now handle all the edge cases:
    if idx == 0 and idx+1 < len(ctx_range):
        return ctx_range[idx+1:]
    if idx > 0 and idx + 1 == len(ctx_range):
        return ctx_range[0:idx]
    if idx > 0 and idx+1 < len(ctx_range):
        return ctx_range[0:idx] + ctx_range[idx+1:]
    """
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
    label = tf.gather(tokens, ctxs)
    feature = tf.fill([tf.size(label)], tokens[target_idx])
    return tf.concat([features, feature], axis=0), \
           tf.concat([labels, label], axis=0), target_idx+1, tokens, window_size


def extract_examples(tokens, window_size, p_num_threads):
    features = tf.constant([], dtype=tf.string)
    labels = tf.constant([], dtype=tf.string)
    target_idx = tf.constant(0, dtype=tf.int32)
    window_size = tf.constant(window_size, dtype=tf.int32)
    max_size = tf.size(tokens, out_type=tf.int32)
    target_idx_less_than_tokens_size = lambda w, x, y, z, k: tf.less(y, max_size)
    result = tf.while_loop(
        cond=target_idx_less_than_tokens_size,
        body=stack_to_features_and_labels,
        loop_vars=[features, labels, target_idx, tokens, window_size],
        shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None]),
                          target_idx.get_shape(), tokens.get_shape(),
                          window_size.get_shape()],
        parallel_iterations=p_num_threads)
    return result[0], result[1]


def get_train_dataset(training_data_filepath, window_size, batch_size,
                      num_epochs, p_num_threads):
    """Generate a Tensorflow Dataset for a Skipgram model."""
    return (tf.data.TextLineDataset(training_data_filepath)
            .map(tf.strings.strip, num_parallel_calls=p_num_threads)
            .filter(lambda x: tf.not_equal(tf.strings.length(x), 0))  # Filter empty strings
            .map(lambda x: tf.strings.split([x]),
                 num_parallel_calls=p_num_threads)
            .map(lambda x: extract_examples(x.values, window_size,
                                            p_num_threads),
                 num_parallel_calls=p_num_threads)
            .flat_map(lambda features, labels: tf.data.Dataset.from_tensor_slices((features, labels)))
            .repeat(num_epochs)
            .batch(batch_size))


def get_model(features, labels, mode, params):
    """Return Word2Vec Skipgram model for a Tensorflow Estimator."""
    with tf.contrib.compiler.jit.experimental_jit_scope():
        vocab = vocab_utils.get_tf_vocab_table(params['word_freq_dict'],
                                               params['min_count'])
        features = vocab.lookup(features)
        labels = vocab.lookup(labels)
        with tf.name_scope('hidden'):
            embeddings = tf.get_variable(
                'embeddings', shape=[params['vocab_size'],
                                     params['embedding_size']],
                initializer=tf.random_uniform_initializer(minval=-1.0,
                                                          maxval=1.0))

        embedded_feat = tf.nn.embedding_lookup(embeddings, features)

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
                               labels=tf.reshape(labels, [-1, 1]),
                               inputs=embedded_feat,
                               num_sampled=params['num_neg_samples'],
                               num_classes=params['vocab_size']))

        men_correlation = params['men'].get_men_correlation(vocab, embeddings)
        metrics = {'MEN': men_correlation}
        tf.summary.scalar('MEN', men_correlation[1])

        with tf.name_scope('optimizer'):
            optimizer = (tf.train.GradientDescentOptimizer(params['learning_rate'])
                         .minimize(loss, global_step=tf.train.get_global_step()))

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer,
                                          eval_metric_ops=metrics)
