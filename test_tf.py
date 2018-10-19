import tensorflow as tf


def convert_to_ex(target_idx, ctx_idx, tokens):
    target = tf.nn.embedding_lookup(tokens, target_idx)
    ctx = tf.nn.embedding_lookup(tokens, ctx_idx)
    # return tf.convert_to_tensor([self._word2id[target.decode('utf8')],
    #                              self._word2id[ctx.decode('utf8')]])
    return tf.convert_to_tensor([0, 1])

def target_idxx(tokens):
    return tf.range(start=0, limit=tf.size(tokens), delta=1)

def ctx_idxx(target_idx, window_size, tokens):
    """
    if target_idx > 0 and target_idx+1 < tf.size(ctx_range):
        return tf.concat([ctx_range[0:target_idx],
                          ctx_range[target_idx+1, tf.size(ctx_range)]], axis=0)
    if target_idx == 0 and target_idx+1 < tf.size(ctx_range):
        return ctx_range[target_idx+1, tf.size(ctx_range)]
    if target_idx > 0 and target_idx+1 == tf.size(ctx_range):
        return ctx_range[0:target_idx]
    return tf.convert_to_tensor([])
    """
    return tf.convert_to_tensor([0, 1])
    return tf.range(start=tf.maximum(0, tf.subtract(target_idx, window_size)), limit=100, delta=1)
    ctx_range = tf.range(start=tf.maximum(0, target_idx-window_size),
                         limit=tf.minimum(tf.size(tokens),
                                          target_idx+window_size+1),
                         delta=1)
    return ctx_range
    t0 = lambda: tf.convert_to_tensor([], dtype=tf.int32)
    t1 = lambda: tf.concat([ctx_range[0:target_idx],
                            ctx_range[target_idx+1:tf.size(ctx_range)]],
                           axis=0)
    t2 = lambda: ctx_range[target_idx+1:tf.size(ctx_range)]
    t3 = lambda: ctx_range[0:target_idx]
    c1 = tf.logical_and(tf.greater(target_idx, 0),
                        tf.logical_not(tf.greater_equal(target_idx+1,
                                                        tf.size(ctx_range))))
    c2 = tf.logical_and(tf.equal(target_idx, 0),
                        tf.logical_not(tf.greater_equal(target_idx+1,
                                                        tf.size(ctx_range))))
    c3 = tf.logical_and(tf.greater(target_idx, 0),
                        tf.equal(target_idx+1, tf.size(ctx_range)))
    return tf.case({c1: t1, c2: t2, c3: t3}, default=t0, exclusive=True)
    #return tf.SparseTensor(indices=[ctx_range], values=t, dense_shape=[1, tf.size(ctx_range)])

def examples(tokens, target_idx, window_size):
    t = tf.map_fn(lambda ctx_idx: convert_to_ex(target_idx, ctx_idx, tokens), ctx_idxx(target_idx, window_size, tokens), infer_shape=False)
    #return tf.contrib.layers.flatten(t)
    #return tf.convert_to_tensor([0,1])
    return t

def extract_examples(tokens, window_size):
    return tf.map_fn(lambda x: tf.map_fn(lambda y: [0, 1], x), tokens)

if __name__ == '__main__':
    tf.enable_eager_execution()
    window_size = 4
    # ctx_idxx(target_idx, window_size, tokens)
    # tf.range(start=0, limit=tf.size(tokens), delta=1)
    with tf.Session(graph=tf.Graph()) as session:
        dataset = (tf.data.TextLineDataset('/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.test')
                   .map(tf.strings.strip)
                   .filter(lambda x: tf.not_equal(tf.strings.length(x), 0))
                   .map(lambda x: tf.strings.split([x]))
                   .map(lambda x: x.values)
                   #.map(lambda tokens: tf.SparseTensor(indices=[1, target_idxx(tokens)], values=tokens, dense_shape=[1])))
                   #.map(lambda tokens: tf.map_fn(lambda target_idx: examples(tokens, target_idx, window_size), target_idxx(tokens), infer_shape=False)))
                   .map(lambda tokens: tf.map_fn(lambda target_idx: tf.map_fn(lambda ctx_idx: convert_to_ex(target_idx, ctx_idx, tokens), ctx_idxx(target_idx, window_size, tokens), dtype=tf.int32), target_idxx(tokens), dtype=tf.int32)))
                   #.map(lambda tokens: extract_examples(tokens, window_size)))
                   #.map(lambda tokens: tf.map_fn(lambda target_idx: tf.map_fn(lambda ctx_idx: tf.convert_to_tensor([0, 1]), tokens, dtype=tf.int32), tokens, dtype=tf.int32)))
        #test_ds = dataset.map(lambda tokens: tf.map_fn(lambda target_idx: target_idx, target_idxx(tokens)))
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        x = iterator.get_next()
        session.run(init_op)
        def test_x(x):
            return x - 2
            #return tf.subtract(x, 2)
            #return tf.maximum(0, tf.subtract(x, 2))
            #return [tf.range(start=tf.maximum(0, tf.subtract(x, 2)), limit=10, delta=1)]
        while True:
            #z = tf.convert_to_tensor([0,1,2,3])
            #y = tf.map_fn(lambda x: tf.range(start=0, limit=tf.size(x, out_type=tf.int64), delta=1, dtype=tf.int64), tf.cast(z, tf.int64))
            #print(y.eval())
            #z = tf.convert_to_tensor([0,1,2])
            #y = tf.map_fn(lambda x: test_x(x), z)
            #print(y.eval())
            #z = tf.range(start=tf.maximum(0, tf.subtract(y, 2)), limit=10, delta=1)
            #print(z.eval())
            #x.infer_shape = False
            print(x)
            print(x[0])
            print(x[0][0])
            print(x.eval())
            #print(x.eval())
            z = tf.reshape(tf.reshape(x, [-1, 2]), [-1, 2])
            t = tf.reshape(z, [-1])
            #print(z)
            #print(t)
            #print(z.eval())
            #print(t.eval())
            #print(x[0].eval(), x[1].eval())
