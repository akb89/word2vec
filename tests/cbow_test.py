import os

import tensorflow as tf
import numpy as np

import nonce2vec.learning.cbow as cbow
import nonce2vec.utils.vocab as vocab_utils

from nonce2vec.models.word2vec import Word2Vec


class CBOWTest(tf.test.TestCase):

    def test_sample_prob(self):
        with self.test_session() as session:
            sampling_rate = 1e-5
            vocab_filepath = os.path.join(os.path.dirname(__file__),
                                          'resources', 'wiki.test.vocab')
            w2v = Word2Vec()
            w2v.load_vocab(vocab_filepath)
            word_freq_table = vocab_utils.get_tf_word_freq_table(w2v._word_count_dict)
            test_data_filepath = os.path.join(os.path.dirname(__file__),
                                              'resources', 'data.txt')
            tf.tables_initializer().run()
            dataset = (tf.data.TextLineDataset(test_data_filepath)
                       .map(tf.strings.strip)
                       .map(lambda x: tf.strings.split([x])))
            iterator = dataset.make_initializable_iterator()
            init_op = iterator.initializer
            x = iterator.get_next()
            session.run(init_op)
            tokens = tf.identity(x.values)
            prob = cbow.sample_prob(tokens, sampling_rate, word_freq_table)
            self.assertAllEqual(
                prob, tf.constant([0.95174104, 0.96964055, 0.9753132,
                                   0.8828317, 0.8525664, 0.9641541, 0.77159685,
                                   0.8885507, 0.9712239, 0.48927504, 0.6388629,
                                   0.89983857, 0.95376116, 0.77159685,
                                   0.70513284, 0.98366046]))

    def test_filter_tokens(self):
        with self.test_session() as session:
            test_data_filepath = os.path.join(os.path.dirname(__file__),
                                              'resources', 'data.txt')
            vocab_filepath = os.path.join(os.path.dirname(__file__),
                                          'resources', 'wiki.test.vocab')
            w2v = Word2Vec()
            w2v.load_vocab(vocab_filepath)
            word_freq = vocab_utils.get_tf_vocab_table(w2v._word_count_dict)
            tf.tables_initializer().run()
            dataset = (tf.data.TextLineDataset(test_data_filepath)
                       .map(tf.strings.strip)
                       .map(lambda x: tf.strings.split([x])))
            iterator = dataset.make_initializable_iterator()
            init_op = iterator.initializer
            x = iterator.get_next()
            session.run(init_op)
            first_tokens = x.values

            second_tokens = x.values
            #print(first_tokens.eval())
            #print(second_tokens.eval())


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

            x = cbow.stack_mean_to_avg_tensor(features, vocab, embeddings)(avg, idx)
            self.assertEqual(x[1].eval(), tf.constant(1).eval())
            self.assertAllEqual(x[0].eval(), tf.constant([[7, 7, 7]]).eval())

            y = cbow.stack_mean_to_avg_tensor(features, vocab, embeddings)(avg, idx+1)
            self.assertEqual(y[1].eval(), tf.constant(2).eval())
            self.assertAllEqual(y[0].eval(),
                                tf.constant([[9.75, 9.75, 9.75]]).eval())

    def test_avg_ctx_features(self):
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
            w = cbow.avg_ctx_features(features, embeddings, vocab,
                                      p_num_threads=1)
            self.assertAllEqual(
                w.eval(), tf.constant([[7, 7, 7], [9.75, 9.75, 9.75],
                                       [9, 9, 9],
                                       [np.nan, np.nan, np.nan]]).eval())


if __name__ == '__main__':
    tf.test.main()
