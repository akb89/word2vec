import os

import tensorflow as tf
import numpy as np

import word2vec.models.word2vec as w2v_model
import word2vec.utils.vocab as vocab_utils

from word2vec.estimators.word2vec import Word2Vec


class W2VEstimatorTest(tf.test.TestCase):

    def test_concat_mean_to_avg_tensor(self):
        with self.session():
            embedding_size = 3
            vocab_size = 4
            batch_size = 2
            embeddings = tf.constant(0, shape=[vocab_size, embedding_size])
            self.assertEqual(embeddings.get_shape()[0], vocab_size)
            self.assertEqual(embeddings.get_shape()[1], embedding_size)
            embeddings = tf.constant([[3., 3., 3.], [6., 6., 6.], [9., 9., 9.],
                                      [12., 12., 12.], [15., 15., 15.]])
            avg = tf.constant(0, shape=[0, embedding_size], dtype=tf.float32)
            idx = tf.constant(0, dtype=tf.int32)

            features = tf.constant([['the', 'first', 'test', '_CBOW#_!MASK_'],
                                    ['the', 'second', 'test', 'out']])
            vocab = tf.lookup.StaticVocabularyTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=['the', 'first', 'second', 'test'],
                    values=[0, 1, 2, 3],
                    key_dtype=tf.string,
                    value_dtype=tf.int64),
                num_oov_buckets=1)
            self.assertAllEqual(vocab.lookup(tf.constant(
                ['the', 'first', 'second', 'test', 'out'])),
                                tf.constant([0, 1, 2, 3, 4]))

            x = w2v_model.concat_mean_to_avg_tensor(features, vocab, embeddings)(avg, idx)
            self.assertEqual(x[1].numpy(), tf.constant(1).numpy())
            self.assertAllEqual(x[0].numpy(), tf.constant([[7, 7, 7]]).numpy())

            y = w2v_model.concat_mean_to_avg_tensor(features, vocab, embeddings)(avg, idx+1)
            self.assertEqual(y[1].numpy(), tf.constant(2).numpy())
            self.assertAllEqual(y[0].numpy(),
                                tf.constant([[9.75, 9.75, 9.75]]).numpy())

    def test_avg_ctx_features_embeddings(self):
        with self.session():
            vocab_size = 4
            batch_size = 4
            embedding_size = 3
            embeddings = tf.constant([[3., 3., 3.], [6., 6., 6.], [9., 9., 9.],
                                      [12., 12., 12.], [15., 15., 15.]])
            features = tf.constant([['the', 'first', 'test', '_CBOW#_!MASK_'],
                                    ['the', 'second', 'test', 'out'],
                                    ['the', 'out', '_CBOW#_!MASK_',
                                     '_CBOW#_!MASK_'],
                                    ['_CBOW#_!MASK_', '_CBOW#_!MASK_',
                                     '_CBOW#_!MASK_', '_CBOW#_!MASK_']])

            vocab = tf.lookup.StaticVocabularyTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=['the', 'first', 'second', 'test'],
                    values=[0, 1, 2, 3],
                    key_dtype=tf.string,
                    value_dtype=tf.int64),
                num_oov_buckets=1)
            w = w2v_model.avg_ctx_features_embeddings(
                features, embeddings, vocab, p_num_threads=1)
            self.assertAllEqual(
                w.numpy(), tf.constant([[7, 7, 7], [9.75, 9.75, 9.75],
                                       [9, 9, 9],
                                       [np.nan, np.nan, np.nan]]).numpy())


if __name__ == '__main__':
    tf.test.main()
