import sys

from collections import defaultdict

import time

import tensorflow as tf


class Word2Vec():

    def __init__(self):
        """Initialize vocab dictionaries."""
        self._vocab = None
        self._word_freq_dict = defaultdict(int)

    @property
    def vocab_size(self):
        return len(self._word_freq_dict)

    def load_vocab(self, vocab_filepath):
        """Load a previously saved vocabulary file."""
        with open(vocab_filepath, 'r', encoding='UTF-8') as vocab_stream:
            for line in vocab_stream:
                word_freq = line.strip().split('\t', 1)
                self._word_freq_dict[word_freq[0]] = int(word_freq[1])

    def _get_tf_vocab_table(self, word_freq_dict, min_count):
        mapping_strings = tf.constant([word for (word, freq) in word_freq_dict.items() if freq >= min_count])
        return tf.contrib.lookup.index_table_from_tensor(
            mapping=mapping_strings, num_oov_buckets=0, default_value=len(word_freq_dict))

    def _generate_train_dataset(self, training_data_filepath, window_size,
                                min_count, batch_size, num_epochs,
                                p_num_threads, shuffling_buffer_size=1,
                                prefetch_batch_size=1,
                                flat_map_pref_batch_size=1):
        # Needs to be here to make sure everything belongs to the same graph
        self._vocab = self._get_tf_vocab_table(self._word_freq_dict, min_count)
        tf.tables_initializer().run()
        def ctx_idxx(target_idx, window_size, tokens):
            """
            # Get the idx corresponding to target-idx in the ctx_range:
            if target_idx - window_size <= 0:
                idx = target_idx
            if target_idx - window_size > 0:
                idx = window_size
            # We would like to return the ctx_range minus the idx, to remove the target:
            return ctx_range[0:idx] + ctx_range[idx+1:]
            # Let us now handle all the edge cases:
            if idx == 0 and idx+1 < len(ctx_range):
                return ctx_range[idx+1:]
            if idx > 0 and idx + 1 == len(ctx_range):
                return ctx_range[0:idx]
            if idx > 0 and idx+1 < len(ctx_range):
                return ctx_range[0:idx] + ctx_range[idx+1:]
            """
            ctx_range = tf.range(start=tf.maximum(tf.constant(0, dtype=tf.int64),
                                                  target_idx-window_size),
                                 limit=tf.minimum(tf.size(tokens, out_type=tf.int64),
                                                  target_idx+window_size+1),
                                 delta=1, dtype=tf.int64)
            idx = tf.case({tf.less_equal(target_idx, window_size): lambda: target_idx,
                           tf.greater(target_idx, window_size): lambda: window_size},
                          exclusive=True)
            t0 = lambda: tf.constant([], dtype=tf.int64)
            t1 = lambda: ctx_range[idx+1:]
            t2 = lambda: ctx_range[0:idx]
            t3 = lambda: tf.concat([ctx_range[0:idx], ctx_range[idx+1:]], axis=0)
            c1 = tf.logical_and(tf.equal(idx, 0),
                                tf.less(idx+1, tf.size(ctx_range, out_type=tf.int64)))
            c2 = tf.logical_and(tf.greater(idx, 0),
                                tf.equal(idx+1, tf.size(ctx_range, out_type=tf.int64)))
            c3 = tf.logical_and(tf.greater(idx, 0),
                                tf.less(idx+1, tf.size(ctx_range, out_type=tf.int64)))
            return tf.case({c1: t1, c2: t2, c3: t3}, default=t0, exclusive=True)

        def stack_to_features_and_labels(features, labels, target_idx, tokens, window_size):
            ctxs = ctx_idxx(target_idx, window_size, tokens)
            #label = tf.nn.embedding_lookup(tokens, ctxs)
            label = tf.gather(tokens, ctxs)
            feature = tf.fill([tf.size(label)], tokens[target_idx])
            return tf.concat([features, feature], axis=0), \
                   tf.concat([labels, label], axis=0), target_idx+1, tokens, window_size

        def extract_examples(tokens, window_size, p_num_threads):
            features = tf.constant([], dtype=tf.int64)
            labels = tf.constant([], dtype=tf.int64)
            target_idx = tf.constant(0, dtype=tf.int64)
            window_size = tf.constant(window_size, dtype=tf.int64)
            max_size = tf.size(tokens, out_type=tf.int64)
            target_idx_less_than_tokens_size = lambda w, x, y, z, k: tf.less(y, max_size)
            result = tf.while_loop(
                cond=target_idx_less_than_tokens_size,
                body=stack_to_features_and_labels,
                loop_vars=[features, labels, target_idx, tokens, window_size],
                shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None]),
                                  target_idx.get_shape(), tokens.get_shape(),
                                  window_size.get_shape()],
                parallel_iterations=p_num_threads)
            return result[0], result[1]
        return (tf.data.TextLineDataset(training_data_filepath)
                .map(tf.strings.strip, num_parallel_calls=p_num_threads)
                .filter(lambda x: tf.not_equal(tf.strings.length(x), 0))  # Filter empty strings
                .map(lambda x: tf.strings.split([x]), num_parallel_calls=p_num_threads)
                .map(lambda x: self._vocab.lookup(x.values), num_parallel_calls=p_num_threads)  # discretize
                .map(lambda tokens: extract_examples(tokens, window_size, p_num_threads), num_parallel_calls=p_num_threads)
                .prefetch(flat_map_pref_batch_size)
                .flat_map(lambda features, labels: tf.data.Dataset.from_tensor_slices((features, labels)))
                .shuffle(buffer_size=shuffling_buffer_size,
                         reshuffle_each_iteration=False)
                .repeat(num_epochs)
                .batch(batch_size)
                .prefetch(prefetch_batch_size))


if __name__ == '__main__':
    TDF = sys.argv[1]
    VOCAB = sys.argv[2]
    PT = int(sys.argv[3])  # preprocessing threads
    FMPBS = int(sys.argv[4])  # flat map prefetch batch size
    BS = int(sys.argv[5])  # batch size
    PBS = int(sys.argv[6])  # prefetching batch size
    NE = int(sys.argv[7])  # num epochs
    SBS = int(sys.argv[8])  # shuffling batch size
    print('-'*80)
    print('RUNNING ON {} THREAD(S) with FMPBS = {}, BS = {}, PBS = {}, NE = {}, SBS = {}'
          .format(PT, FMPBS, BS, PBS, NE, SBS))
    tf.enable_eager_execution()
    w2v = Word2Vec()
    w2v.load_vocab(VOCAB)
    WIN = 5  # window size
    MINC = 1  # min count
    with tf.Session(graph=tf.Graph()) as session:
        dataset = w2v._generate_train_dataset(TDF, WIN, MINC, BS, NE, PT, SBS,
                                              PBS, FMPBS)
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        x = iterator.get_next()
        i = 1
        session.run(init_op)
        start = time.monotonic()
        while True:
            try:
                session.run(x)
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
