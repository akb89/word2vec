import os

import tensorflow as tf

import nonce2vec.utils.datasets as datasets_utils
import nonce2vec.utils.vocab as vocab_utils

from nonce2vec.estimators.word2vec import Word2Vec


class DatasetsUtilsTest(tf.test.TestCase):

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
            prob = datasets_utils.sample_prob(tokens, sampling_rate,
                                              word_freq_table)
            self.assertAllEqual(
                prob, tf.constant([0.95174104, 0.96964055, 0.9753132,
                                   0.8828317, 0.8525664, 0.9641541, 0.77159685,
                                   0.8885507, 0.9712239, 0.48927504, 0.6388629,
                                   0.89983857, 0.95376116, 0.77159685,
                                   0.70513284, 0.98366046]))

    def test_filter_tokens_mask(self):
        with self.test_session() as session:
            min_count = 50
            sampling_rate = 1.
            test_data_filepath = os.path.join(os.path.dirname(__file__),
                                              'resources', 'data.txt')
            vocab_filepath = os.path.join(os.path.dirname(__file__),
                                          'resources', 'wiki.test.vocab')
            w2v = Word2Vec()
            w2v.load_vocab(vocab_filepath)
            word_count_table = vocab_utils.get_tf_word_count_table(w2v._word_count_dict)
            word_freq_table = vocab_utils.get_tf_word_freq_table(w2v._word_count_dict)
            tf.tables_initializer().run()
            dataset = (tf.data.TextLineDataset(test_data_filepath)
                       .map(tf.strings.strip)
                       .map(lambda x: tf.strings.split([x])))
            iterator = dataset.make_initializable_iterator()
            init_op = iterator.initializer
            x = iterator.get_next()
            session.run(init_op)
            self.assertAllEqual(datasets_utils.filter_tokens_mask(
                x.values, min_count, sampling_rate, word_count_table,
                word_freq_table), tf.constant([
                    True, True, True, False, False, True, False, False, True,
                    False, False, False, True, False, False, True]))

    def test_extract_cbow_examples(self):
        with self.test_session():
            window_size = 2
            p_num_threads = 1
            tokens = tf.constant(['this', 'is', 'a', 'test', 'of', 'sent'])
            features, labels = datasets_utils.extract_examples(
                tokens, 'cbow', window_size, p_num_threads)
            self.assertAllEqual(
                labels,
                tf.constant([[b'this'], [b'is'], [b'a'], [b'test'], [b'of'],
                             [b'sent']]))
            self.assertAllEqual(
                features,
                tf.constant([[b'is', b'a', b'_CBOW#_!MASK_', b'_CBOW#_!MASK_'],
                             [b'this', b'a', b'test', b'_CBOW#_!MASK_'],
                             [b'this', b'is', b'test', b'of'],
                             [b'is', b'a', b'of', b'sent'],
                             [b'a', b'test', b'sent', b'_CBOW#_!MASK_'],
                             [b'test', b'of', b'_CBOW#_!MASK_', b'_CBOW#_!MASK_']]))

    def test_get_cbow_train_dataset(self):
        with self.test_session() as session:
            sampling_rate = 1.
            window_size = 5
            min_count = 50
            batch_size = 2
            num_epochs = 1
            p_num_threads = 1
            shuffling_buffer_size = 1
            vocab_filepath = os.path.join(os.path.dirname(__file__),
                                          'resources', 'wiki.test.vocab')
            w2v = Word2Vec()
            w2v.load_vocab(vocab_filepath)
            test_data_filepath = os.path.join(os.path.dirname(__file__),
                                              'resources', 'data.txt')
            dataset = datasets_utils.get_w2v_train_dataset(
                test_data_filepath, 'cbow', w2v._word_count_dict,
                window_size, min_count, sampling_rate, batch_size, num_epochs,
                p_num_threads, shuffling_buffer_size)
            tf.tables_initializer().run()
            iterator = dataset.make_initializable_iterator()
            init_op = iterator.initializer
            x = iterator.get_next()
            session.run(init_op)
            features, labels = session.run(x)
            self.assertAllEqual(
                features, tf.constant([[b'is', b'a', b'that', b'-', b'on',
                                        b'_CBOW#_!MASK_', b'_CBOW#_!MASK_',
                                        b'_CBOW#_!MASK_', b'_CBOW#_!MASK_',
                                        b'_CBOW#_!MASK_'],
                                       [b'anarchism', b'a', b'that', b'-',
                                        b'on', b'.', b'_CBOW#_!MASK_',
                                        b'_CBOW#_!MASK_', b'_CBOW#_!MASK_',
                                        b'_CBOW#_!MASK_']]))
            self.assertAllEqual(labels, tf.constant([[b'anarchism'], [b'is']]))


if __name__ == '__main__':
    tf.test.main()
