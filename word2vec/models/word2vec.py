"""Return a Word2Vec model (CBOW or SKIPGRAM) for a Tensorflow Estimator."""

import logging

import math
import tensorflow as tf

import word2vec.utils.vocab as vocab_utils

logger = logging.getLogger(__name__)

__all__ = ('model')


def concat_mean_to_avg_tensor(features, vocab, embeddings):
    """Concatenate the mean of ctx embeddings to the features tensor."""
    def internal_func(avg, idx):
        # For a given set of features, corresponding to a given set
        # of context words
        feat_row = features[idx]
        # select only valid context words
        is_valid_string = tf.not_equal(feat_row, '_CBOW#_!MASK_')
        valid_feats = tf.boolean_mask(tensor=feat_row, mask=is_valid_string)
        # discretized the features
        discretized_feats = vocab.lookup(valid_feats)
        # select their corresponding embeddings
        embedded_feats = tf.nn.embedding_lookup(params=embeddings,
                                                ids=discretized_feats)
        # average over the given context word embeddings
        mean = tf.reduce_mean(input_tensor=embedded_feats, axis=0)
        # concatenate to the return averaged tensor stacking all
        # averaged context embeddings for a given batch
        avg = tf.concat([avg, [mean]], axis=0)
        return [avg, idx+1]
    return internal_func


def avg_ctx_features_embeddings(features, embeddings, vocab, p_num_threads):
    """Average context embeddings."""
    feat_batch_size = features.get_shape()[0]
    embedding_size = embeddings.get_shape()[1]
    idx_within_batch_size = lambda x, idx: tf.less(idx, feat_batch_size)
    concat_func = concat_mean_to_avg_tensor(features, vocab, embeddings)
    avg = tf.constant([], shape=[0, embedding_size], dtype=tf.float32)
    idx = tf.constant(0, dtype=tf.int32)
    avg, idx = tf.while_loop(
        cond=idx_within_batch_size,
        body=concat_func,
        loop_vars=[avg, idx],
        shape_invariants=[tf.TensorShape([None, embedding_size]),
                          idx.get_shape()],
        parallel_iterations=p_num_threads,
        maximum_iterations=feat_batch_size)
    return avg


def _model(features, labels, mode, params):
    vocab_table = vocab_utils.get_tf_vocab_table(params['words'])
    with tf.compat.v1.name_scope('hidden'):
        embeddings = tf.compat.v1.get_variable(
            'embeddings', shape=[params['vocab_size'],
                                 params['embedding_size']],
            initializer=tf.compat.v1.random_uniform_initializer(minval=-1.0,
                                                                maxval=1.0))
    if params['mode'] == 'cbow':
        discret_labels = vocab_table.lookup(labels)
        discret_features_embeddings = avg_ctx_features_embeddings(
            features, embeddings, vocab_table, params['p_num_threads'])
    elif params['mode'] == 'skipgram':
        discret_labels = vocab_table.lookup(labels)
        discret_features_embeddings = tf.nn.embedding_lookup(
            params=embeddings, ids=vocab_table.lookup(features))

    with tf.compat.v1.name_scope('weights'):
        nce_weights = tf.compat.v1.get_variable(
            'nce_weights', shape=[params['vocab_size'],
                                  params['embedding_size']],
            initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=1.0 / math.sqrt(params['embedding_size'])))

    with tf.compat.v1.name_scope('biases'):
        nce_biases = tf.compat.v1.get_variable(
            'nce_biases', shape=[params['vocab_size']],
            initializer=tf.compat.v1.zeros_initializer)

    with tf.compat.v1.name_scope('loss'):
        loss = tf.reduce_mean(
            input_tensor=tf.nn.nce_loss(weights=nce_weights,
                                        biases=nce_biases,
                                        labels=discret_labels,
                                        inputs=discret_features_embeddings,
                                        num_sampled=params['num_neg_samples'],
                                        num_classes=params['vocab_size']))

    with tf.compat.v1.name_scope('optimizer'):
        optimizer = (tf.compat.v1.train.GradientDescentOptimizer(
            params['learning_rate']).minimize(
                loss, global_step=tf.compat.v1.train.get_global_step()))
    men_correlation = params['men'].get_men_correlation(
        vocab_table, embeddings)
    metrics = {'MEN': men_correlation}
    tf.compat.v1.summary.scalar('MEN', men_correlation[1])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer,
                                      eval_metric_ops=metrics)


def model(features, labels, mode, params):
    """Return the model_fn function of a TF Estimator for a Word2Vec model."""
    if params['mode'] not in ['cbow', 'skipgram']:
        raise Exception('Unsupported Word2Vec mode \'{}\''
                        .format(params['mode']))
    if params['xla']:
        with tf.compiler.jit.experimental_jit_scope():
            return _model(features, labels, mode, params)
    else:
        return _model(features, labels, mode, params)
