import sys

import time

import tensorflow as tf

import nonce2vec.utils.datasets as datasets_utils
import nonce2vec.utils.vocab as vocab_utils
from nonce2vec.estimators.word2vec import Word2Vec

if __name__ == '__main__':
    TDF = sys.argv[1]
    VOCAB = sys.argv[2]
    WIN = int(sys.argv[3])  # window size
    MINC = int(sys.argv[4])  # min count
    SAMP = float(sys.argv[5])  # sampling rate
    BS = int(sys.argv[6])  # batch size
    NE = int(sys.argv[7])  # epochs
    PT = int(sys.argv[8])  # preprocessing threads
    SBS = int(sys.argv[9])  # shuffling batch size
    MODE = sys.argv[10]
    print('-'*80)
    print('RUNNING {} ON {} THREAD(S) with WIN = {}, MINC = {}, SAMP = {}, '
          'BS = {}, NE = {}, SBS = {}'.format(MODE, PT, WIN, MINC, SAMP, BS,
                                              NE, SBS))
    tf.enable_eager_execution()
    w2v = Word2Vec()
    print('loading vocab...')
    w2v.load_vocab(VOCAB, MINC)
    print('done loading vocab')
    with tf.Session(graph=tf.Graph()) as session:
        dataset = datasets_utils.get_w2v_train_dataset(
            TDF, MODE, w2v._words, w2v._counts, w2v._total_count, WIN, SAMP, BS, NE, PT, SBS)
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        print('initializing tables...')
        word_count_table = vocab_utils.get_tf_word_count_table(w2v._words, w2v._counts)
        tf.tables_initializer().run()
        print('done initializing tables')
        print('generating datasets...')

        x = iterator.get_next()
        i = 1
        session.run(init_op)
        start = time.monotonic()
        while True:
            try:
                #print('step {}'.format(i))
                features, labels = session.run(x)
                #print(features)
                #print(labels)
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
