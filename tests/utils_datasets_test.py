import os

import tensorflow as tf
import math

import word2vec.utils.datasets as datasets_utils
import word2vec.utils.vocab as vocab_utils

from word2vec.estimators.word2vec import Word2Vec


class DatasetsUtilsTest(tf.test.TestCase):

    def test_skipgram_concat_to_features_and_labels(self):
        with self.session():
            tf.compat.v1.disable_tensor_equality()
            train_mode = 'skipgram'
            window_size = 3
            tokens = tf.constant(['this', 'is', 'a', 'test', 'sent'])
            features = tf.constant([], dtype=tf.string)
            labels = tf.constant([], shape=[0, 1], dtype=tf.string)
            idx = tf.constant(0)
            features, labels, idx = \
             datasets_utils.concat_to_features_and_labels(
                 tokens, train_mode, window_size)(features, labels, idx)
            self.assertAllEqual(labels, tf.constant(
                [[b'is'], [b'a'], [b'test']]))
            self.assertAllEqual(features, tf.constant(
                [b'this', b'this', b'this']))
            features, labels, _ = \
             datasets_utils.concat_to_features_and_labels(
                 tokens, train_mode, window_size)(features, labels, idx)
            self.assertAllEqual(labels, tf.constant(
                [[b'is'], [b'a'], [b'test'], [b'this'], [b'a'], [b'test'], [b'sent']]))
            self.assertAllEqual(features, tf.constant(
                [b'this', b'this', b'this', b'is', b'is', b'is', b'is']))


    def test_sample_prob(self):
        with self.session() as session:
            sampling_rate = 1e-5
            min_count = 1
            vocab_filepath = os.path.join(os.path.dirname(__file__),
                                          'resources', 'wiki.test.vocab')
            w2v = Word2Vec()
            w2v.load_vocab(vocab_filepath, min_count)
            word_count_table = vocab_utils.get_tf_word_count_table(
                w2v._words, w2v._counts)
            test_data_filepath = os.path.join(os.path.dirname(__file__),
                                              'resources', 'data.txt')
            dataset = (tf.data.TextLineDataset(test_data_filepath)
                       .map(tf.strings.strip)
                       .map(lambda x: tf.strings.split([x]).to_sparse()))
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            x = iterator.get_next()
            tokens = tf.convert_to_tensor(value=x.values.numpy())
            prob = datasets_utils.sample_prob(
                tokens, sampling_rate, word_count_table, w2v._total_count)
            sample = lambda x: 1 - math.sqrt(sampling_rate / (x / w2v._total_count))
            self.assertAllEqual(
                prob, tf.constant(
                    [sample(y) for y in word_count_table.lookup(tokens).numpy()],
                    dtype=tf.float64))

    def test_filter_tokens_mask(self):
        with self.session() as session:
            min_count = 50
            sampling_rate = 1.
            test_data_filepath = os.path.join(os.path.dirname(__file__),
                                              'resources', 'data.txt')
            vocab_filepath = os.path.join(os.path.dirname(__file__),
                                          'resources', 'wiki.test.vocab')
            w2v = Word2Vec()
            w2v.load_vocab(vocab_filepath, min_count)
            word_count_table = vocab_utils.get_tf_word_count_table(
                w2v._words, w2v._counts)
            dataset = (tf.data.TextLineDataset(test_data_filepath)
                       .map(tf.strings.strip)
                       .map(lambda x: tf.strings.split([x]).to_sparse()))
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            x = iterator.get_next()
            self.assertAllEqual(datasets_utils.filter_tokens_mask(
                x.values, sampling_rate, word_count_table, w2v._total_count),
                                tf.constant(
                                    [True, True, True, False, False, True,
                                     False, False, True, False, False, False,
                                     True, False, False, True]))

    def test_sample_tokens(self):
        with self.session() as session:
            min_count = 50
            sampling_rate = 1.
            test_data_filepath = os.path.join(os.path.dirname(__file__),
                                              'resources', 'data.txt')
            vocab_filepath = os.path.join(os.path.dirname(__file__),
                                          'resources', 'wiki.test.vocab')
            w2v = Word2Vec()
            w2v.load_vocab(vocab_filepath, min_count)
            word_count_table = vocab_utils.get_tf_word_count_table(
                w2v._words, w2v._counts)
            dataset = (tf.data.TextLineDataset(test_data_filepath)
                       .map(tf.strings.strip)
                       .map(lambda x: tf.strings.split([x]).to_sparse()))
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            x = iterator.get_next()
            self.assertAllEqual(datasets_utils.sample_tokens(
                x.values, sampling_rate, word_count_table, w2v._total_count),
                                tf.constant([b'anarchism', b'is', b'a',
                                             b'that', b'-', b'on', b'.']))

    def test_extract_cbow_examples(self):
        with self.session():
            tf.compat.v1.disable_tensor_equality()
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
        with self.session() as session:
            tf.compat.v1.disable_tensor_equality()
            # tf.compat.v1.disable_eager_execution()
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
            w2v.load_vocab(vocab_filepath, min_count)
            test_data_filepath = os.path.join(os.path.dirname(__file__),
                                              'resources', 'data.txt')
            dataset = datasets_utils.get_w2v_train_dataset(
                test_data_filepath, 'cbow', w2v._words, w2v._counts,
                w2v._total_count, window_size, sampling_rate, batch_size,
                num_epochs, p_num_threads, shuffling_buffer_size)
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            features, labels = iterator.get_next()
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
