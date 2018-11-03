"""Utilities to handle vocabulary and discretization in Tensorflow."""

import tensorflow as tf

__all__ = ('get_tf_vocab_table', 'get_tf_word_count_table',
           'get_tf_word_freq_table')


def get_tf_vocab_table(word_count_dict, min_count):
    """Return a TF lookup.index_table built from word_freq_dict.keys().

    Using min_count helps filtering out rare words and makes for a hidden
    layer of a lower dimension.
    """
    mapping_strings = tf.constant([
        word for (word, count) in word_count_dict.items()
        if count >= min_count])
    with tf.name_scope('vocab'):
        return tf.contrib.lookup.index_table_from_tensor(
            mapping=mapping_strings, num_oov_buckets=0,
            default_value=len(word_count_dict))


def get_tf_word_count_table(word_count_dict, min_count):
    """Convert a python OrderedDict to a TF HashTable mapping words to
    their respective corpus counts.
    Using min_count limits the size of the table
    """
    with tf.name_scope('word_count'):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                [word for word, count in word_count_dict.items()
                 if count >= min_count],
                [count for word, count in word_count_dict.items()
                 if count >= min_count]),
            default_value=0)


def get_tf_word_freq_table(word_count_dict, min_count):
    """Convert a python OrderedDict to a TF HashTable mapping words to
    their respective corpus frequencies.
    Using min_count limits the size of the table
    """
    total_count = sum(count for count in word_count_dict.values())
    with tf.name_scope('word_count'):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                [word for word, count in word_count_dict.items()
                 if count >= min_count],
                [count/total_count for word, count in word_count_dict.items()
                 if count >= min_count]),
            default_value=0)
