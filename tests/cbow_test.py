import tensorflow as tf

import nonce2vec.learning.cbow as cbow


class CBOWTest(tf.test.TestCase):

    def test_stack_mean_to_avg_tensor(self):
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

            x = cbow.stack_mean_to_avg_tensor(vocab)(features, avg, idx,
                                                     embeddings)
            self.assertAllEqual(x[0].eval(), features.eval())
            self.assertEqual(x[2].eval(), tf.constant(1).eval())
            self.assertAllEqual(x[3].eval(), embeddings.eval())
            self.assertAllEqual(x[1].eval(), tf.constant([[7, 7, 7]]).eval())

            y = cbow.stack_mean_to_avg_tensor(vocab)(features, avg, idx+1,
                                                     embeddings)
            self.assertAllEqual(y[0].eval(), features.eval())
            self.assertEqual(y[2].eval(), tf.constant(2).eval())
            self.assertAllEqual(y[3].eval(), embeddings.eval())
            self.assertAllEqual(y[1].eval(),
                                tf.constant([[9.75, 9.75, 9.75]]).eval())

    def test_avg_ctx_features(self):
        with self.test_session():
            vocab_size = 4
            embeddings = tf.constant([[3., 3., 3.], [6., 6., 6.], [9., 9., 9.],
                                      [12., 12., 12.], [15., 15., 15.]])
            features = tf.constant([['the', 'first', 'test', '_CBOW#_!MASK_'],
                                    ['the', 'second', 'test', 'out']])

            vocab = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(['the', 'first', 'second', 'test']),
                num_oov_buckets=0, default_value=vocab_size)
            tf.tables_initializer().run()
            w = cbow.avg_ctx_features(features, embeddings, vocab,
                                      p_num_threads=1)
            self.assertAllEqual(
                w.eval(), tf.constant([[7, 7, 7], [9.75, 9.75, 9.75]]).eval())


if __name__ == '__main__':
    tf.test.main()
