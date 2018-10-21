"""Welcome to Nonce2Vec.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

import scipy
import tensorflow as tf

#from nonce2vec.models.word2vec import Word2Vec
from nonce2vec.models.word2vec_estimator import Word2Vec

import nonce2vec.utils.config as cutils
import nonce2vec.utils.files as futils

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)

# Note: this is scipy's spearman, without tie adjustment
def _spearman(x, y):
    return scipy.stats.spearmanr(x, y)[0]


def _get_men_pairs_and_sim(men_dataset):
    pairs = []
    humans = []
    with open(men_dataset, 'r') as men_stream:
        for line in men_stream:
            items = line.rstrip('\n').split()
            pairs.append((items[0], items[1]))
            humans.append(float(items[2]))
    return pairs, humans


def _check_men(args):
    logger.info('Checking embeddings quality against MEN similarity ratings')
    pairs, humans = _get_men_pairs_and_sim(args.men_dataset)
    logger.info('Loading word2vec model from {}'.format(args.model_dirpath))
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(os.path.join(args.model_dirpath, 'model.meta'))
        saver.restore(session, tf.train.latest_checkpoint(args.model_dirpath))
        graph = tf.get_default_graph()
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])
        embeddings = graph.get_tensor_by_name('embeddings/Variable:0')
    # logger.info('Model loaded')
    # system_actual = []
    # human_actual = []  # This is needed because we may not be able to
    #                    # calculate cosine for all pairs
    # count = 0
    # for (first, second), human in zip(pairs, humans):
    #     if first not in model.wv.vocab or second not in model.wv.vocab:
    #         logger.error('Could not find one of more pair item in model '
    #                      'vocabulary: {}, {}'.format(first, second))
    #         continue
    #     sim = _cosine_similarity(model.wv[first], model.wv[second])
    #     system_actual.append(sim)
    #     human_actual.append(human)
    #     count += 1
    # spr = _spearman(human_actual, system_actual)
    # logger.info('SPEARMAN: {} calculated over {} items'.format(spr, count))


def _train(args):
    logger.info('Training Tensorflow implementation of Word2Vec')
    output_model_dirpath = futils.get_model_dirpath(args.datafile,
                                                    args.outputdir,
                                                    args.train_mode,
                                                    args.alpha, args.neg,
                                                    args.window, args.sample,
                                                    args.epochs,
                                                    args.min_count, args.size)
    w2v = Word2Vec(output_model_dirpath)
    if not args.vocab or (args.vocab and not os.path.exists(args.vocab)):
        if not args.datafile:
            raise Exception(
                'Unspecified data_filepath. You need to specify the data '
                'file from which to build the vocabulary, or to specify a '
                'valid vocabulary filepath')
        if not os.path.exists(args.vocab):
            logger.warning('The specified vocabulary filepath does not seem '
                           'to exist: {}'.format(args.vocab))
            logger.warning('Re-building vocabulary from scratch')
        vocab_filepath = futils.get_vocab_filepath(args.datafile,
                                                   output_model_dirpath)
        w2v.build_vocab(args.datafile, vocab_filepath)
    else:
        w2v.load_vocab(args.vocab)
    w2v.train(args.train_mode, args.datafile, output_model_dirpath,
              args.min_count, args.batch, args.size, args.neg, args.alpha,
              args.window, args.epochs, args.sample, args.p_num_threads,
              args.t_num_threads)
    #w2v.evaluate()


def main():
    """Launch Nonce2Vec."""
    parser = argparse.ArgumentParser(prog='nonce2vec')
    subparsers = parser.add_subparsers()
    # a shared set of parameters when using gensim
    parser_gensim = argparse.ArgumentParser(add_help=False)
    parser_gensim.add_argument('--p-num-threads', type=int, default=1,
                               help='number of threads used for preprocessing')
    parser_gensim.add_argument('--t-num-threads', type=int, default=1,
                               help='number of threads used for training')
    parser_gensim.add_argument('--alpha', type=float,
                               help='initial learning rate')
    parser_gensim.add_argument('--neg', type=int,
                               help='number of negative samples')
    parser_gensim.add_argument('--window', type=int,
                               help='window size')
    parser_gensim.add_argument('--sample', type=float,
                               help='subsampling rate')
    parser_gensim.add_argument('--epochs', type=int,
                               help='number of epochs')
    parser_gensim.add_argument('--min-count', type=int,
                               help='min frequency count')

    # train word2vec with gensim from a wikipedia dump
    parser_train = subparsers.add_parser(
        'train', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_gensim],
        help='generate pre-trained embeddings from wikipedia dump via '
             'a Tensorflow implementation of Word2Vec')
    parser_train.set_defaults(func=_train)
    parser_train.add_argument('--data', required=True, dest='datafile',
                              help='absolute path to training data file')
    parser_train.add_argument('--size', type=int, default=400,
                              help='vector dimensionality')
    parser_train.add_argument('--batch', type=int, default=128,
                              help='batch size')
    parser_train.add_argument('--train-mode', choices=['cbow', 'skipgram'],
                              help='how to train word2vec')
    parser_train.add_argument('--outputdir', required=True,
                              help='Absolute path to outputdir to save model')
    parser_train.add_argument('--vocab',
                              help='Absolute path to the the vocabulary file.'
                                   'Where to load and/or save the vocabulary')

    parser_check = subparsers.add_parser(
        'check', formatter_class=argparse.RawTextHelpFormatter,
        help='check w2v embeddings quality by calculating correlation with '
             'the similarity ratings in the MEN dataset.')
    parser_check.set_defaults(func=_check_men)
    parser_check.add_argument('--data', required=True, dest='men_dataset',
                              help='absolute path to dataset')
    parser_check.add_argument('--model', required=True, dest='model_dirpath',
                              help='absolute path to the directory where the '
                                   'Tensorflow model data are stored')
    args = parser.parse_args()
    args.func(args)
