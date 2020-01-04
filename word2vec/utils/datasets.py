"""A set of utils to generate Tensorflow Dataset instances."""

import logging

import tensorflow as tf

import word2vec.utils.vocab as vocab_utils

logger = logging.getLogger(__name__)

__all__ = ('get_w2v_train_dataset')


def ctx_idxx(target_idx, window_size, tokens):
    """Return positions of context words."""
    ctx_range = tf.range(start=tf.maximum(tf.constant(0, dtype=tf.int32),
                                          target_idx-window_size),
                         limit=tf.minimum(tf.size(input=tokens, out_type=tf.int32),
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
                        tf.less(idx+1, tf.size(input=ctx_range, out_type=tf.int32)))
    c2 = tf.logical_and(tf.greater(idx, 0),
                        tf.equal(idx+1, tf.size(input=ctx_range, out_type=tf.int32)))
    c3 = tf.logical_and(tf.greater(idx, 0),
                        tf.less(idx+1, tf.size(input=ctx_range, out_type=tf.int32)))
    return tf.case({c1: t1, c2: t2, c3: t3}, default=t0, exclusive=True)


# pylint: disable=R1710
def concat_to_features_and_labels(tokens, train_mode, window_size):
    """Concatenate features and labels into Tensor."""
    def internal_func(features, labels, target_idx):
        if train_mode not in ['cbow', 'skipgram']:
            raise Exception('Unsupported Word2Vec mode \'{}\''
                            .format(train_mode))
        ctxs = ctx_idxx(target_idx, window_size, tokens)
        if train_mode == 'cbow':
            ctx_features = tf.gather(tokens, ctxs)
            diff = 2 * window_size - tf.size(input=ctx_features)
            ctx_features = tf.reshape(ctx_features, [1, -1])
            paddings = tf.concat(
                [tf.constant([[0, 0]]),
                 tf.concat([tf.constant([[0]]), [[diff]]], axis=1)], axis=0)
            padded_ctx_features = tf.pad(tensor=ctx_features, paddings=paddings,
                                         constant_values='_CBOW#_!MASK_')
            label = tf.reshape(tokens[target_idx], [1, -1])
            return tf.concat([features, padded_ctx_features], axis=0), \
                   tf.concat([labels, label], axis=0), target_idx+1
        if train_mode == 'skipgram':
            label = tf.reshape(tf.gather(tokens, ctxs), [-1, 1])
            feature = tf.fill([tf.size(input=label)], tokens[target_idx])
            return tf.concat([features, feature], axis=0), \
                   tf.concat([labels, label], axis=0), target_idx+1
    return internal_func


def extract_examples(tokens, train_mode, window_size, p_num_threads):
    """Extract (features, labels) examples from a list of tokens."""
    if train_mode not in ['cbow', 'skipgram']:
        raise Exception('Unsupported Word2Vec mode \'{}\''
                        .format(train_mode))
    if train_mode == 'cbow':
        features = tf.constant([], shape=[0, 2*window_size], dtype=tf.string)
    elif train_mode == 'skipgram':
        features = tf.constant([], dtype=tf.string)
    labels = tf.constant([], shape=[0, 1], dtype=tf.string)
    target_idx = tf.constant(0, dtype=tf.int32)
    concat_func = concat_to_features_and_labels(tokens, train_mode,
                                                window_size)
    max_size = tf.size(input=tokens, out_type=tf.int32)
    idx_below_tokens_size = lambda w, x, idx: tf.less(idx, max_size)
    if train_mode == 'cbow':
        result = tf.while_loop(
            cond=idx_below_tokens_size,
            body=concat_func,
            loop_vars=[features, labels, target_idx],
            shape_invariants=[tf.TensorShape([None, 2*window_size]),
                              tf.TensorShape([None, 1]),
                              target_idx.get_shape()],
            parallel_iterations=p_num_threads)
    elif train_mode == 'skipgram':
        result = tf.while_loop(
            cond=idx_below_tokens_size,
            body=concat_func,
            loop_vars=[features, labels, target_idx],
            shape_invariants=[tf.TensorShape([None]),
                              tf.TensorShape([None, 1]),
                              target_idx.get_shape()],
            parallel_iterations=p_num_threads)
    return result[0], result[1]


def sample_prob(tokens, sampling_rate, word_count_table, total_count):
    """Sample according to w2v paper formula: p = 1 - sqrt(t/f)."""
    freq = word_count_table.lookup(tokens) / total_count
    return 1 - tf.sqrt(sampling_rate / freq)


def filter_tokens_mask(tokens, sampling_rate, word_count_table, total_count):
    """Filter tokens in a sentence.

    Remove unfrequent words (words with count < min_count) and apply
    subsampling according to the original W2V paper.
    The word_count_table already contains words with counts >= min_count
    and its default value is 0, hence the tf.greater(..., 0) condition.
    """
    return tf.logical_and(
        tf.greater(word_count_table.lookup(tokens),
                   tf.constant(0, dtype=tf.float64)),
        tf.less(sample_prob(tokens, sampling_rate, word_count_table,
                            total_count),
                tf.random.uniform(shape=[tf.size(input=tokens)],
                                  minval=0, maxval=1, dtype=tf.float64)))


def sample_tokens(tokens, sampling_rate, word_count_table, total_count):
    """Apply subsampling to a set of tokens."""
    return tf.boolean_mask(
        tensor=tokens, mask=filter_tokens_mask(
            tokens, sampling_rate, word_count_table, total_count))


def get_w2v_train_dataset(training_data_filepath, train_mode,
                          words, counts, total_count, window_size,
                          sampling_rate, batch_size, num_epochs,
                          p_num_threads, shuffling_buffer_size):
    """Generate a Tensorflow Dataset for a Word2Vec model."""
    word_count_table = vocab_utils.get_tf_word_count_table(words, counts)
    return (tf.data.TextLineDataset(training_data_filepath)
            .map(tf.strings.strip, num_parallel_calls=p_num_threads)
            .filter(lambda x: tf.not_equal(tf.strings.length(input=x), 0))
            .map(lambda x: tf.strings.split([x]).to_sparse(),
                 num_parallel_calls=p_num_threads)
            .map(lambda x: sample_tokens(x.values, sampling_rate,
                                         word_count_table, total_count),
                 num_parallel_calls=p_num_threads)
            # Keep examples with at least 2 tokens
            .filter(lambda x: tf.greater(tf.size(input=x), 1))
            .map(lambda x: extract_examples(x, train_mode, window_size,
                                            p_num_threads),
                 num_parallel_calls=p_num_threads)
            .flat_map(lambda features, labels: \
                      tf.data.Dataset.from_tensor_slices((features, labels)))
            .shuffle(buffer_size=shuffling_buffer_size,
                     reshuffle_each_iteration=False)
            .repeat(num_epochs)
            .batch(batch_size, drop_remainder=True))
            # we need drop_remainder to statically know the batch dimension
            # this is required to get features.get_shape()[0] in
            # w2v_estimator.avg_ctx_features_embeddings
