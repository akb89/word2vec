import os

import tensorflow as tf

from word2vec.evaluation.men import MEN


class MENEvaluationTest(tf.test.TestCase):

    def test_get_men_correlation(self):
        with self.session():
            men = MEN(os.path.join(os.path.dirname(__file__), 'resources',
                                   'men_test.txt'))
            vocab = tf.lookup.StaticVocabularyTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=['one', 'two', 'three', 'four', 'five', 'six'],
                    values=[0, 1, 2, 3, 4, 5],
                    key_dtype=tf.string,
                    value_dtype=tf.int64),
                num_oov_buckets=1)
            embeddings = tf.constant([[0., 1.], [0., 1.], [1., 1.],
                                      [0., 1.], [0., 1.], [1., 0.]])
            normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
            self.assertAllEqual(normalized_embeddings.numpy(),
                                tf.constant([[0., 1.], [0., 1.],
                                             [0.70710677, 0.70710677],
                                             [0., 1.], [0., 1.], [1., 0.]]))
            left_label_embeddings = tf.nn.embedding_lookup(
                params=normalized_embeddings,
                ids=vocab.lookup(tf.constant(men.left_labels, dtype=tf.string)))
            self.assertAllEqual(
                left_label_embeddings.numpy(),
                tf.constant([[0., 1.], [0.70710677, 0.70710677],
                             [0., 1.]]).numpy())
            right_label_embeddings = tf.nn.embedding_lookup(
                params=normalized_embeddings,
                ids=vocab.lookup(tf.constant(men.right_labels, dtype=tf.string)))
            self.assertAllEqual(
                right_label_embeddings.numpy(),
                tf.constant([[0., 1.], [0., 1.], [1., 0.]]).numpy())
            sim_predictions = 1 - tf.compat.v1.losses.cosine_distance(
                left_label_embeddings, right_label_embeddings, axis=1,
                reduction=tf.compat.v1.losses.Reduction.NONE)
            self.assertAllEqual(sim_predictions.numpy(),
                                tf.constant([[1.], [0.70710677], [0.]]))
            men_correlation = men.get_men_correlation(vocab, embeddings)
            self.assertGreater(men_correlation[1].numpy(), 0.97)


if __name__ == '__main__':
    tf.test.main()
