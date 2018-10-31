"""Evaluation with the MEN dataset."""

import tensorflow as tf

__all__ = ('MEN')


class MEN():
    """A class to store MEN examples and process correlations in TF."""

    def __init__(self, men_filepath):
        """Load MEN labels and similarity metrics from file."""
        with open(men_filepath, 'r') as men_stream:
            self._left_labels = []
            self._right_labels = []
            self._sim_values = []
            for line in men_stream:
                tokens = line.strip().split()
                self._left_labels.append(tokens[0])
                self._right_labels.append(tokens[1])
                self._sim_values.append(float(tokens[2]))

    @property
    def left_labels(self):
        """Return all MEN left word labels as a list of strings."""
        return self._left_labels

    @property
    def right_labels(self):
        """Return all MEN right word labels as a list of strings."""
        return self._right_labels

    @property
    def sim_values(self):
        """Return all MEN similarity values as a list of floats."""
        return self._sim_values

    def get_men_correlation(self, vocab, embeddings):
        """Return spearman correlation metric on the MEN dataset."""
        with tf.contrib.compiler.jit.experimental_jit_scope():
            normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
            left_label_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings,
                vocab.lookup(tf.constant(self.left_labels, dtype=tf.string)))
            right_label_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings,
                vocab.lookup(tf.constant(self.right_labels, dtype=tf.string)))
            sim_predictions = tf.losses.cosine_distance(
                left_label_embeddings, right_label_embeddings, axis=1,
                reduction=tf.losses.Reduction.NONE)
            return tf.contrib.metrics.streaming_pearson_correlation(
                sim_predictions, tf.constant(self.sim_values,
                                             dtype=tf.float32))
