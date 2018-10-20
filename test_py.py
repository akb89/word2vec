import tensorflow as tf

def get_ctx_items(tokens, target_id, window_size):
    ctx_1 = tokens[max(0, target_id-window_size):target_id]
    ctx_2 = tokens[target_id+1:min(len(tokens), target_id+window_size+1)]
    return [*ctx_1, *ctx_2]

def get_example():
    training_data_filepath = '/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.test'
    window_size = 15
    with open(training_data_filepath, 'r') as training_datastream:
        for line in training_datastream:
            tokens = line.strip().split()
            for target_id, target in enumerate(tokens):
                for ctx in get_ctx_items(tokens, target_id, window_size):
                    #yield self._word2id[target], self._word2id[ctx]
                    yield 0, 1

if __name__ == '__main__':
    window_size = 15
    training_data_filepath = '/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.all.utf8.sent.split.lower'
    # for item in get_example(training_data_filepath, window_size):
    #     print(item)
    with tf.Session(graph=tf.Graph()) as session:
        dataset = (tf.data.Dataset.from_generator(get_example,
                                               (tf.int32, tf.int32))
                .shuffle(buffer_size=10,
                         reshuffle_each_iteration=False)
                .repeat(5)
                .batch(128)
                .prefetch(10))
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        x = iterator.get_next()
        session.run(init_op)
        while True:
            print(x[0].eval(), x[1].eval())
