import os

import tensorflow as tf

import word2vec.utils.vocab as vocab_utils
from word2vec.estimators.word2vec import Word2Vec


class VocabUtilsTest(tf.test.TestCase):

    def test_get_tf_vocab_table(self):
        with self.test_session():
            min_count = 1
            vocab_filepath = os.path.join(os.path.dirname(__file__),
                                          'resources', 'wiki.test.vocab')
            w2v = Word2Vec()
            w2v.load_vocab(vocab_filepath, min_count)
            vocab = vocab_utils.get_tf_vocab_table(w2v._words)
            tf.compat.v1.tables_initializer().run()
            self.assertAllEqual(
                vocab.lookup(tf.constant(['anarchism', 'is', 'UKN@!',
                                          '1711'])),
                tf.constant([0, 1, len(w2v._words),
                             len(w2v._words)-1]))

    def test_get_word_count_table(self):
        with self.test_session():
            min_count = 1
            vocab_filepath = os.path.join(os.path.dirname(__file__),
                                          'resources', 'wiki.test.vocab')
            w2v = Word2Vec()
            w2v.load_vocab(vocab_filepath, min_count)
            word_count = vocab_utils.get_tf_word_count_table(
                w2v._words, w2v._counts)
            tf.compat.v1.tables_initializer().run()
            self.assertAllEqual(
                word_count.lookup(tf.constant(['anarchism', 'is', 'UKN@!',
                                               '1711'])),
                tf.constant([112, 283, 0, 1]))
            freq = word_count.lookup(
                tf.constant(['anarchism', 'is', 'UKN@!', '1711'])) / w2v._total_count
            self.assertAllEqual(
                freq, tf.constant([112/26084, 283/26084, 0, 1/26084],
                                  dtype=tf.float64))


if __name__ == '__main__':
    tf.test.main()
