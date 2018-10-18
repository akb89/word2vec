import tensorflow as tf


def extract_examples(tokens):
    examples = []
    for token in tokens:
        examples.append((96, 69))
    return tf.convert_to_tensor(examples)

def extract_ex(target, context):
    return (self._word2id[target], self._word2id[context])

def get_target_idxx(split_sparse_tensor_value):
    return tf.range(start=0, limit=tf.size(split_sparse_tensor_value), delta=1)

def get_examples(target, target_idx, split_sparse_tensor_value):
    ctxx = tf.nn.embedding_lookup(split_sparse_tensor_value, get_ctx_idxx(split_sparse_tensor_value, target_idx))
    return Tensor([(target, ctx), (target, ctx), ...])

def convert_to_ex(target_idx, ctx_idx, tokens):
    target = tf.nn.embedding_lookup(tokens, target_idx)
    ctx = tf.nn.embedding_lookup(tokens, ctx_idx)
    # return tf.convert_to_tensor([self._word2id[target.decode('utf8')],
    #                              self._word2id[ctx.decode('utf8')]])
    return tf.convert_to_tensor([0, 1])

def target_idxx(tokens):
    return tf.range(start=0, limit=tf.size(tokens), delta=1)

def ctx_idxx(target_idx, window_size, tokens):
    ctx_range = tf.range(start=tf.minimum(0, target_idx-window_size),
                         limit=tf.maximum(tf.size(tokens),
                                          target_idx+window_size+1),
                         delta=1)
    if tf.equal(ctx_range[0], target_idx):
        return ctx_range[target_idx+1, tf.size(ctx_range)]
    if tf.equal(ctx_range[-1], target_idx):
        return ctx_range[0:target_idx]
    return tf.concat([ctx_range[0:target_idx],
                      ctx_range[target_idx+1, tf.size(ctx_range)]], axis=0)

if __name__ == '__main__':
    tf.enable_eager_execution()
    window_size = 2
    # ctx_idxx(target_idx, window_size, tokens)
    # tf.range(start=0, limit=tf.size(tokens), delta=1)
    with tf.Session(graph=tf.Graph()) as session:
        dataset = (tf.data.TextLineDataset('/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.test')
                   .map(tf.strings.strip)
                   .filter(lambda x: tf.not_equal(tf.strings.length(x), 0))
                   .map(lambda x: tf.strings.split([x]))
                   .map(lambda x: x.values)
                   #.map(lambda tokens: tf.map_fn(lambda target: tf.map_fn(lambda ctx: convert_to_ex(target, ctx, tokens), tokens, dtype=tf.int32), tokens, dtype=tf.int32)))
                   .map(lambda tokens: tf.map_fn(lambda target_idx: tf.map_fn(lambda ctx_idx: convert_to_ex(target_idx, ctx_idx, tokens), ctx_idxx(target_idx, window_size, tokens), dtype=tf.int32), target_idxx(tokens), dtype=tf.int32)))
                   #.map(lambda tokens: tf.map_fn(lambda target_and_idx: tf.map_fn(lambda ctx: convert_to_ex(target_and_idx[0], ctx), (tokens, tf.range(start=0, limit=tf.size(tokens), delta=1)), dtype=tf.int32), tokens, dtype=tf.int32)))
                   #.map(lambda x: tf.map_fn(extract_examples, x)))
                   #.apply(tf.map_fn(lambda x: extract_examples, x.values)))
        #dataset = dataset.map(lambda x: tf.map_fn(lambda y: y, x))
        #dataset.map(lambda x: tf.map_fn(lambda y: tf.map_fn(lambda z: z, x), x))
        #dataset.map(lambda tokens: tf.map_fn(lambda target: tf.map_fn(lambda ctx: convert_to_ex(target, ctx), tokens), tokens))
        # dataset.map(lambda x: tf.map_fn(lambda y: get_examples(y[0], y[1], x), (x, get_targets_idx(x))))
        # dataset.map(lambda x: tf.map_fn(lambda y: tf.map_fn(lambda z: extract_ex(y, z), get_contexts()), x))
        # targets_idx = get_targets_idx(x)
        # ctx_idx_range = tf.range(start=y_idx-window, limit=y_idx+window+1, delta=1).filter(lambda x: x != y_idx and x >= 0 and x <= tf.size(x))
        # tf.range(start=x_range-window, limit=len(x), delta=1).filter()
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        x = iterator.get_next()
        session.run(init_op)
        while True:
            y = x.eval()
            print(y)
            # print(y[0:2])
            # print(y[3:4])
            # print(tf.concat([y[0:2], y[3:4]], axis=0).eval())
            #z = tf.stack(y[0:1], y[1:2]).eval()
            #print(z)
