"""Utilities to handle vocabulary and discretization in Tensorflow."""

import tensorflow as tf

__all__ = ('get_tf_vocab_table', 'get_tf_word_count_table')


def get_tf_vocab_table(words):
    """Return a TF lookup.index_table built from a list of words.

    Words are already prefiltered so that word count >= min_count.
    """
    with tf.compat.v1.name_scope('vocab'):
        return tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=words,
                values=list(range(0, len(words))),
                key_dtype=tf.string,
                value_dtype=tf.int64),
            num_oov_buckets=1)


def get_tf_word_count_table(words, counts):
    """Return  a TF HashTable mapping words to their respective corpus counts.

    Words and counts are already prefiltered so that word count >= min_count.
    Type tf.float64 is necessary as counts get later divided to obtain
    frequencies.
    """
    with tf.compat.v1.name_scope('word_count'):
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                words, counts, value_dtype=tf.float64),
            default_value=0)
