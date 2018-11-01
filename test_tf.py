import os

import tensorflow as tf
from collections import OrderedDict

import nonce2vec.utils.vocab as vocab_utils

from nonce2vec.evaluation.men import MEN


if __name__ == '__main__':
    vocab_filepath = '/Users/AKB/Github/nonce2vec/models/wiki.test.vocab'
    with tf.Session(graph=tf.Graph()) as session:
        diff = 2 * 2 - tf.size(tf.constant(2))
        #print(k)
        #print(k.eval())

        t = tf.constant([[1, 2, 3]])
        #print(t.eval())
        paddings = tf.constant([[0, 0], [0, 2]])
        #print(paddings.eval())
        m = tf.pad(t, paddings, 'CONSTANT')
        #print(m.eval())

        feat = tf.constant([], shape=[0, 2*5], dtype=tf.string)
        ctx_features = tf.constant(['this', 'is', 'a'], dtype=tf.string)
        x = tf.reshape(ctx_features, [1, -1])
        #print(x.eval())
        paddings = tf.concat([tf.constant([[0, 0]]), tf.concat([tf.constant([[0]]), [[diff]]], axis=1)], axis=0)

        z = tf.pad(x, paddings, constant_values='_CBOW#_!MASK_')
        #print(z.eval())

        t = tf.constant([], shape=[0, 6], dtype=tf.string)
        r = tf.concat([t, z], axis=0)
        features = tf.concat([r, z], axis=0)
        #print(features.eval())

        labels = tf.constant([], dtype=tf.string)
        tokens = tf.constant(['ta', 'rg', 'et'])
        labels = tf.concat([labels, [tokens[0]]], axis=0)
        labels = tf.concat([labels, [tokens[1]]], axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(2)

        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        x = iterator.get_next()
        i = 1
        session.run(init_op)

        word_freq_dict = OrderedDict()

        with open(vocab_filepath, 'r', encoding='UTF-8') as vocab_stream:
            for line in vocab_stream:
                word_freq = line.strip().split('\t', 1)
                word_freq_dict[word_freq[0]] = int(word_freq[1])

        vocab_size = len(word_freq_dict) + 1
        embedding_size = 4
        batch_size = 2
        vocab = vocab_utils.get_tf_vocab_table(word_freq_dict, 1)

        embeddings = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                     initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        def stack_mean_to_avg_tensor(features, avg, idx, embeddings):
            # For a given set of features, corresponding to a given set
            # of context words
            feat_row = features[idx]
            # select only valid context words
            is_valid_string = tf.not_equal(feat_row, '_CBOW#_!MASK_')
            valid_feats = tf.boolean_mask(feat_row, is_valid_string)
            # discretized the features
            discretized_feats = vocab.lookup(valid_feats)
            # select their corresponding embeddings
            embedded_feats = tf.nn.embedding_lookup(embeddings, discretized_feats)
            # average over the given context word embeddings
            mean = tf.reduce_mean(embedded_feats, 0)
            # concatenate to the return averaged tensor stacking all
            # averaged context embeddings for a given batch
            avg = tf.concat([avg, [mean]], axis=0)
            return features, avg, idx+1, embeddings
        tf.tables_initializer().run()
        tf.initialize_all_variables().run()
        x_emb = tf.constant([[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4], [8, 8, 8, 8]])
        x_idxx = tf.constant([[0, 1, 2], [3]])
        x_idx = x_idxx[0]
        print(x_idx.eval())
        x_lu = tf.nn.embedding_lookup(x_emb, x_idx)
        x_mean = tf.reduce_mean(x_lu, 0)
        print(x_lu.eval())
        print(x_mean.eval())
        while True:
            try:
                #y = session.run(x)
                y = x
                features = y[0]
                labels = y[1]
                p_num_threads = 1
                idx = tf.constant(0, dtype=tf.int32)
                print(features.eval())
                print(batch_size)

                avg = tf.constant([], shape=[0, embedding_size], dtype=tf.float32)

                idx_within_batch_size = lambda v, w, x, y: tf.less(x, batch_size)
                result = tf.while_loop(
                    cond=idx_within_batch_size,
                    body=stack_mean_to_avg_tensor,
                    loop_vars=[features, avg, idx, embeddings],
                    shape_invariants=[features.get_shape(),
                                      tf.TensorShape([None, embedding_size]),
                                      idx.get_shape(),
                                      embeddings.get_shape()],
                    parallel_iterations=p_num_threads)
                print(result[3].eval())
            except tf.errors.OutOfRangeError:
                break
