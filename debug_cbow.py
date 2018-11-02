import tensorflow as tf
import math
import numpy as np
import nonce2vec.learning.cbow as cbow
import nonce2vec.utils.vocab as vutils

from nonce2vec.models.word2vec import Word2Vec

if __name__ == '__main__':
    vocab_filepath = '/Users/AKB/Github/nonce2vec/models/wiki.test.vocab'
    train_filepath = '/Users/AKB/Github/nonce2vec/data/wikipedia/wiki.test'
    tf.enable_eager_execution()
    with tf.Session(graph=tf.Graph()) as session:
        window_size = 5
        batch_size = 128
        num_epochs = 1
        p_num_threads = 4
        shuffling_buffer_size = 1
        embedding_size = 128
        min_count = 1
        num_neg_samples = 5
        learning_rate = 0.025
        dataset = cbow.get_train_dataset(
            train_filepath, window_size, batch_size, num_epochs, p_num_threads,
            shuffling_buffer_size)
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        x = iterator.get_next()
        session.run(init_op)
        step = 1
        w2v = Word2Vec()
        w2v.load_vocab(vocab_filepath)
        vocab = vutils.get_tf_vocab_table(w2v._word_count_dict)
        embeddings = tf.get_variable(
            'embeddings', shape=[w2v.vocab_size,
                                 embedding_size],
            initializer=tf.random_uniform_initializer(minval=-1.0,
                                                      maxval=1.0))
        nce_weights = tf.get_variable(
            'nce_weights', shape=[w2v.vocab_size, embedding_size],
            initializer=tf.truncated_normal_initializer(
                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.get_variable(
            'nce_biases', shape=[w2v.vocab_size],
            initializer=tf.zeros_initializer)
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()
        while True:
            try:
                features, labels = x[0], x[1]
                #if step > 16500:
                discretized_labels = tf.reshape(vocab.lookup(labels), [-1, 1])
                discretized_avg_features = cbow.avg_ctx_features(
                    features, embeddings, vocab, p_num_threads)
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=nce_weights,
                                   biases=nce_biases,
                                   labels=discretized_labels,
                                   inputs=discretized_avg_features,
                                   num_sampled=num_neg_samples,
                                   num_classes=w2v.vocab_size))
                optimizer = (tf.train.GradientDescentOptimizer(learning_rate)
                             .minimize(loss, global_step=tf.train.get_global_step()))
                loss_eval = loss.eval()
                print('step {} -- loss = {}'.format(step, loss_eval))
                optimizer.run()
                if np.isnan(loss_eval):
                    print(labels.eval(), discretized_labels.eval())
                    print(features.eval(), discretized_avg_features.eval())
                step += 1
            except tf.errors.OutOfRangeError:
                break
