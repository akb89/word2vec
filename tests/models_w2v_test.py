import os

import tensorflow as tf
import numpy as np

import word2vec.models.word2vec as w2v_model
import word2vec.utils.vocab as vocab_utils

from word2vec.estimators.word2vec import Word2Vec


class W2VEstimatorTest(tf.test.TestCase):

    def test_concat_mean_to_avg_tensor(self):
        with self.test_session():
            embedding_size = 3
            vocab_size = 4
            batch_size = 2
            embeddings = tf.constant([], shape=[vocab_size, embedding_size])
            self.assertEqual(embeddings.get_shape()[0], vocab_size)
            self.assertEqual(embeddings.get_shape()[1], embedding_size)
            embeddings = tf.constant([[3., 3., 3.], [6., 6., 6.], [9., 9., 9.],
                                      [12., 12., 12.], [15., 15., 15.]])
            avg = tf.constant([], shape=[0, embedding_size], dtype=tf.float32)
            idx = tf.constant(0, dtype=tf.int32)

            features = tf.constant([['the', 'first', 'test', '_CBOW#_!MASK_'],
                                    ['the', 'second', 'test', 'out']])

            vocab = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(['the', 'first', 'second', 'test']),
                num_oov_buckets=0, default_value=vocab_size)
            tf.tables_initializer().run()
            self.assertAllEqual(vocab.lookup(tf.constant(
                ['the', 'first', 'second', 'test', 'out'])),
                                tf.constant([0, 1, 2, 3, 4]))

            x = w2v_model.concat_mean_to_avg_tensor(features, vocab, embeddings)(avg, idx)
            self.assertEqual(x[1].eval(), tf.constant(1).eval())
            self.assertAllEqual(x[0].eval(), tf.constant([[7, 7, 7]]).eval())

            y = w2v_model.concat_mean_to_avg_tensor(features, vocab, embeddings)(avg, idx+1)
            self.assertEqual(y[1].eval(), tf.constant(2).eval())
            self.assertAllEqual(y[0].eval(),
                                tf.constant([[9.75, 9.75, 9.75]]).eval())

    def test_avg_ctx_features_embeddings(self):
        with self.test_session():
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

            vocab = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(['the', 'first', 'second', 'test']),
                num_oov_buckets=0, default_value=vocab_size)
            tf.tables_initializer().run()
            w = w2v_model.avg_ctx_features_embeddings(
                features, embeddings, vocab, p_num_threads=1)
            self.assertAllEqual(
                w.eval(), tf.constant([[7, 7, 7], [9.75, 9.75, 9.75],
                                       [9, 9, 9],
                                       [np.nan, np.nan, np.nan]]).eval())


if __name__ == '__main__':
    tf.test.main()
