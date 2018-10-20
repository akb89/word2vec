import tensorflow as tf

def ctx_idxx(target_idx, window_size, tokens):
    """
    if target_idx > 0 and target_idx+1 < tf.size(ctx_range):
        return tf.concat([ctx_range[0:target_idx],
                          ctx_range[target_idx+1, tf.size(ctx_range)]], axis=0)
    if target_idx == 0 and target_idx+1 < tf.size(ctx_range):
        return ctx_range[target_idx+1, tf.size(ctx_range)]
    if target_idx > 0 and target_idx+1 == tf.size(ctx_range):
        return ctx_range[0:target_idx]
    return tf.constant([])
    """
    ctx_range = tf.range(start=tf.maximum(0, target_idx-window_size),
                         limit=tf.minimum(tf.size(tokens),
                                          target_idx+window_size+1),
                         delta=1)
    t0 = lambda: tf.constant([], dtype=tf.int32)
    t1 = lambda: tf.concat([ctx_range[0:target_idx],
                            ctx_range[target_idx+1:tf.size(ctx_range)]],
                           axis=0)
    t2 = lambda: ctx_range[target_idx+1:tf.size(ctx_range)]
    t3 = lambda: ctx_range[0:target_idx]
    c1 = tf.logical_and(tf.greater(target_idx, 0),
                        tf.less(target_idx+1, tf.size(ctx_range)))
    c2 = tf.logical_and(tf.equal(target_idx, 0),
                        tf.less(target_idx+1, tf.size(ctx_range)))
    c3 = tf.logical_and(tf.greater(target_idx, 0),
                        tf.equal(target_idx+1, tf.size(ctx_range)))
    return tf.case({c1: t1, c2: t2, c3: t3}, default=t0, exclusive=True)

def stack_to_features_and_labels(features, labels, target_idx, tokens):
    ctxs = ctx_idxx(target_idx, 2, tokens)
    label = tf.nn.embedding_lookup(tokens, ctxs)
    feature = tf.fill([tf.size(label)], tokens[target_idx])
    return tf.concat([features, feature], axis=0), \
           tf.concat([labels, label], axis=0), target_idx+1, tokens

def extract_examples(tokens, window_size, p_num_threads):
    features = tf.constant([], dtype=tf.int32)
    labels = tf.constant([], dtype=tf.int32)
    target_idx = tf.constant(0)
    target_idx_less_than_tokens_size = lambda w, x, y, z: tf.less(y, tf.size(tokens))
    result = tf.while_loop(
        cond=target_idx_less_than_tokens_size,
        body=stack_to_features_and_labels,
        loop_vars=[features, labels, target_idx, tokens],
        shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None]),
                          target_idx.get_shape(), tokens.get_shape()],
        parallel_iterations=p_num_threads)
    return result[0], result[1]

if __name__ == '__main__':
    tf.enable_eager_execution()
    window_size = 4
    p_num_threads= 2
    with tf.Session(graph=tf.Graph()) as session:
        dataset = (tf.data.TextLineDataset('/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.test')
                   .map(tf.strings.strip)
                   .filter(lambda x: tf.not_equal(tf.strings.length(x), 0))  # Filter empty strings
                   .map(lambda x: tf.strings.split([x]))
                   .map(lambda x: x.values)
                   .map(lambda tokens: tf.map_fn(lambda token: 0, tokens, dtype=tf.int32))  # discretized
                   .map(lambda tokens: extract_examples(tokens, window_size, p_num_threads))
                   .flat_map(lambda features, labels: tf.data.Dataset.from_tensor_slices((features, labels))))
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        x = iterator.get_next()
        session.run(init_op)
        while True:
            print(x[0].eval(), x[1].eval())
