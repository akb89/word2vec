import os

import tensorflow as tf

from word2vec.evaluation.men import MEN


class MENEvaluationTest(tf.test.TestCase):

    def test_get_men_correlation(self):
        with self.test_session():
            men = MEN(os.path.join(os.path.dirname(__file__), 'resources',
                                   'men_test.txt'))
            vocab = tf.lookup.index_table_from_tensor(
                mapping=tf.constant(['one', 'two', 'three', 'four', 'five',
                                     'six']),
                num_oov_buckets=0, default_value=6)
            tf.compat.v1.tables_initializer().run()
            embeddings = tf.constant([[0., 1.], [0., 1.], [1., 1.],
                                      [0., 1.], [0., 1.], [1., 0.]])
            normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
            self.assertAllEqual(normalized_embeddings.eval(),
                                tf.constant([[0., 1.], [0., 1.],
                                             [0.70710677, 0.70710677],
                                             [0., 1.], [0., 1.], [1., 0.]]))
            left_label_embeddings = tf.nn.embedding_lookup(
                params=normalized_embeddings,
                ids=vocab.lookup(tf.constant(men.left_labels, dtype=tf.string)))
            self.assertAllEqual(
                left_label_embeddings.eval(),
                tf.constant([[0., 1.], [0.70710677, 0.70710677],
                             [0., 1.]]).eval())
            right_label_embeddings = tf.nn.embedding_lookup(
                params=normalized_embeddings,
                ids=vocab.lookup(tf.constant(men.right_labels, dtype=tf.string)))
            self.assertAllEqual(
                right_label_embeddings.eval(),
                tf.constant([[0., 1.], [0., 1.], [1., 0.]]).eval())
            sim_predictions = 1 - tf.compat.v1.losses.cosine_distance(
                left_label_embeddings, right_label_embeddings, axis=1,
                reduction=tf.compat.v1.losses.Reduction.NONE)
            self.assertAllEqual(sim_predictions.eval(),
                                tf.constant([[1.], [0.70710677], [0.]]))
            men_correlation = men.get_men_correlation(vocab, embeddings)
            tf.compat.v1.global_variables_initializer().run()
            tf.compat.v1.local_variables_initializer().run()
            self.assertGreater(men_correlation[1].eval(), 0.97)


if __name__ == '__main__':
    tf.test.main()
