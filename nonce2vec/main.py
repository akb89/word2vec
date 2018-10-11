"""Welcome to Nonce2Vec.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

from nonce2vec.models.word2vec import Word2Vec

import nonce2vec.utils.config as cutils
import nonce2vec.utils.files as futils

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _train(args):
    logger.info('Training Tensorflow implementation of Word2Vec')
    w2v = Word2Vec(min_count=args.min_count, batch_size=args.batch,
                   embedding_size=args.size, num_neg_samples=args.neg,
                   learning_rate=args.alpha, window_size=args.window,
                   num_epochs=args.epochs, subsampling_rate=args.sample,
                   num_threads=args.num_threads)
    vocab_filepath = futils.get_vocab_filepath(args.datafile, args.min_count,
                                               args.outputdir)
    if args.vocab:
        logger.info('Re-using pre-existing vocabulary file {}'
                    .format(args.vocab))
        w2v.load_vocab(args.vocab)
    else:
        w2v.build_vocab(args.datafile, vocab_filepath)
    batches_filepath = futils.get_batches_filepath(args.datafile, args.batch,
                                                   args.outputdir)
    if args.batches:
        logger.info('Re-using pre-existing batches file {}'
                    .format(args.batches))
    else:
        w2v.generate_batches(args.datafile, batches_filepath)
    w2v.initialize_tf_graph()
    w2v.train(batches_filepath)


def main():
    """Launch Nonce2Vec."""
    parser = argparse.ArgumentParser(prog='nonce2vec')
    subparsers = parser.add_subparsers()
    # a shared set of parameters when using gensim
    parser_gensim = argparse.ArgumentParser(add_help=False)
    parser_gensim.add_argument('--num_threads', type=int, default=1,
                               help='number of threads to be used by gensim')
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
    parser_gensim.add_argument('--min_count', type=int,
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
    parser_train.add_argument('--train_mode', choices=['cbow', 'skipgram'],
                              help='how to train word2vec')
    parser_train.add_argument('--outputdir', required=True,
                              help='Absolute path to outputdir to save model')
    parser_train.add_argument('--vocab',
                              help='Absolute path to saved vocabulary file')
    parser_train.add_argument('--batches',
                              help='Absolute path to saved batches file')
    args = parser.parse_args()
    args.func(args)
