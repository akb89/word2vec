"""Utilities to handle vocabulary and discretization in Tensorflow."""

import tensorflow as tf

__all__ = ('get_tf_vocab_table')


def get_tf_vocab_table(word_freq_dict, min_count):
    """Return a TF lookup.index_table built from the word_freq_dict."""
    mapping_strings = tf.constant(
        [word for (word, freq) in word_freq_dict.items() if freq >= min_count])
    with tf.name_scope('vocab'):
        return tf.contrib.lookup.index_table_from_tensor(
            mapping=mapping_strings, num_oov_buckets=0,
            default_value=len(word_freq_dict))
