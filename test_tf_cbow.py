import sys

from collections import defaultdict

import time

import tensorflow as tf


def ctx_idxx(target_idx, window_size, tokens):
    ctx_range = tf.range(start=tf.maximum(tf.constant(0, dtype=tf.int32),
                                          target_idx-window_size),
                         limit=tf.minimum(tf.size(tokens, out_type=tf.int32),
                                          target_idx+window_size+1),
                         delta=1, dtype=tf.int32)
    idx = tf.case({tf.less_equal(target_idx, window_size): lambda: target_idx,
                   tf.greater(target_idx, window_size): lambda: window_size},
                  exclusive=True)
    t0 = lambda: tf.constant([], dtype=tf.int32)
    t1 = lambda: ctx_range[idx+1:]
    t2 = lambda: ctx_range[0:idx]
    t3 = lambda: tf.concat([ctx_range[0:idx], ctx_range[idx+1:]], axis=0)
    c1 = tf.logical_and(tf.equal(idx, 0),
                        tf.less(idx+1, tf.size(ctx_range, out_type=tf.int32)))
    c2 = tf.logical_and(tf.greater(idx, 0),
                        tf.equal(idx+1, tf.size(ctx_range, out_type=tf.int32)))
    c3 = tf.logical_and(tf.greater(idx, 0),
                        tf.less(idx+1, tf.size(ctx_range, out_type=tf.int32)))
    return tf.case({c1: t1, c2: t2, c3: t3}, default=t0, exclusive=True)


def stack_to_features_and_labels(features, labels, target_idx, tokens,
                                 window_size):
    ctxs = ctx_idxx(target_idx, window_size, tokens)
    ctx_features = tf.gather(tokens, ctxs)
    diff = 2 * window_size - tf.size(ctx_features)
    ctx_features = tf.reshape(ctx_features, [1, -1])
    paddings = tf.concat([tf.constant([[0, 0]]), tf.concat([tf.constant([[0]]), [[diff]]], axis=1)], axis=0)
    padded_ctx_features = tf.pad(ctx_features, paddings,
                                 constant_values='_CBOW#_!MASK_')
    label = tokens[target_idx]
    return tf.concat([features, padded_ctx_features], axis=0), \
           tf.concat([labels, [label]], axis=0), target_idx+1, tokens, \
           window_size


def extract_examples(tokens, window_size, p_num_threads):
    features = tf.constant([], shape=[0, 2*window_size], dtype=tf.string)
    labels = tf.constant([], dtype=tf.string)
    target_idx = tf.constant(0, dtype=tf.int32)
    t_window_size = tf.constant(window_size, dtype=tf.int32)
    max_size = tf.size(tokens, out_type=tf.int32)
    target_idx_less_than_tokens_size = lambda w, x, y, z, k: tf.less(y, max_size)
    result = tf.while_loop(
        cond=target_idx_less_than_tokens_size,
        body=stack_to_features_and_labels,
        loop_vars=[features, labels, target_idx, tokens, t_window_size],
        shape_invariants=[tf.TensorShape([None, 2*window_size]), tf.TensorShape([None]),
                          target_idx.get_shape(), tokens.get_shape(),
                          t_window_size.get_shape()],
        parallel_iterations=p_num_threads)
    return result[0], result[1]
    #return tf.data.Dataset.from_tensor_slices((result[0], result[1]))


def get_train_dataset(training_data_filepath, window_size, batch_size,
                      num_epochs, p_num_threads):
    """Generate a Tensorflow Dataset for a Skipgram model."""
    return (tf.data.TextLineDataset(training_data_filepath)
            .map(tf.strings.strip, num_parallel_calls=p_num_threads)
            .filter(lambda x: tf.not_equal(tf.strings.length(x), 0))  # Filter empty strings
            .map(lambda x: tf.strings.split([x]),
                 num_parallel_calls=p_num_threads)
            .map(lambda x: extract_examples(x.values, window_size,
                                            p_num_threads),
                 num_parallel_calls=p_num_threads)
            .flat_map(lambda features, labels: tf.data.Dataset.from_tensor_slices((features, labels)))
            .batch(batch_size))


if __name__ == '__main__':
    TDF = sys.argv[1]
    VOCAB = sys.argv[2]
    PT = int(sys.argv[3])  # preprocessing threads
    BS = int(sys.argv[4])  # batch size
    NE = int(sys.argv[5])  # num epochs
    print('-'*80)
    print('RUNNING ON {} THREAD(S) with BS = {}, NE = {}'.format(PT, BS, NE))
    tf.enable_eager_execution()
    # w2v = Word2Vec()
    # w2v.load_vocab(VOCAB)
    WIN = 5  # window size
    MINC = 1  # min count
    with tf.Session(graph=tf.Graph()) as session:
        dataset = get_train_dataset(TDF, WIN, BS, NE, PT)
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        x = iterator.get_next()
        i = 1
        session.run(init_op)
        start = time.monotonic()
        while True:
            try:
                y = session.run(x)
                is_valid_string = tf.not_equal(y[0], '_CBOW#_!MASK_')
                print(y[0], y[1])
                print(is_valid_string.eval())
                features = tf.boolean_mask(y[0], is_valid_string)
                print(features.eval())
                i += 1
            except tf.errors.OutOfRangeError:
                end = time.monotonic()
                total = round(end-start, 2)
                print('Processed {} batches of size {} in {}s'.format(i, BS, total))
                average_batch_s = round((end - start) / i)
                average_batch_ms = round(((end - start) / i) * 1000, 2)
                print('Average time per batch = {}s or {}ms'
                      .format(average_batch_s, average_batch_ms))
                average_ex_ms = round(((end - start) / (i * BS)) * 1000, 2)
                average_ex_mus = round(((end - start) / (i * BS)) * 1000000)
                print('Average time per example = {}ms or {}µs'
                      .format(average_ex_ms, average_ex_mus))
                ex_per_line = WIN * .85 * 50  # a simple heuristic to get the number of ex. per line depending on the window size
                total_num_lines = 124302571
                total_num_ex = total_num_lines * ex_per_line
                average_ex_s = (end - start) / (i * BS)
                total_wiki_h = (average_ex_s * total_num_ex) / 3600
                total_wiki_d = total_wiki_h / 24
                print('EPT on full Wikipedia dump, per epoch = '
                      '{} day(s) or {}h'.format(round(total_wiki_d), round(total_wiki_h)))
                print('-'*80)
                break
