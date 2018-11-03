"""Utilities to handle vocabulary and discretization in Tensorflow."""

import tensorflow as tf

__all__ = ('get_tf_vocab_table', 'get_tf_word_count_table')


def get_tf_vocab_table(words):
    """Return a TF lookup.index_table built from a list of words.

    Words are already prefiltered so that word count >= min_count.
    """
    with tf.name_scope('vocab'):
        return tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.convert_to_tensor(words), num_oov_buckets=0,
            default_value=len(words))


def get_tf_word_count_table(words, counts):
    """Return  a TF HashTable mapping words to their respective corpus counts.

    Words and counts are already prefiltered so that word count >= min_count.
    Type tf.float64 is necessary as counts get later divided to obtain
    frequencies.
    """
    with tf.name_scope('word_count'):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                words, counts, value_dtype=tf.float64),
            default_value=0)
