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
    output_model_dirpath = futils.get_model_dirpath(args.datafile,
                                                    args.outputdir,
                                                    args.train_mode,
                                                    args.alpha, args.neg,
                                                    args.window, args.sample,
                                                    args.epochs,
                                                    args.min_count, args.size,
                                                    args.batch)
    w2v = Word2Vec()
    if not args.vocab or (args.vocab and not os.path.exists(args.vocab)):
        if not args.datafile:
            raise Exception(
                'Unspecified data_filepath. You need to specify the data '
                'file from which to build the vocabulary, or to specify a '
                'valid vocabulary filepath')
        if args.vocab and not os.path.exists(args.vocab):
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
              args.t_num_threads, args.save_summary_steps,
              args.save_checkpoints_steps, args.keep_checkpoint_max,
              args.log_step_count_steps)


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
    parser_train.add_argument('--save-summary-steps', type=int,
                              default=100000, help='')
    parser_train.add_argument('--save-checkpoints-steps', type=int,
                              default=1000000, help='')
    parser_train.add_argument('--keep-checkpoint-max', type=int,
                              default=3, help='')
    parser_train.add_argument('--log-step-count-steps', type=int,
                              default=100000, help='')
    parser_train.add_argument('--train-mode', choices=['cbow', 'skipgram'],
                              help='how to train word2vec')
    parser_train.add_argument('--outputdir', required=True,
                              help='absolute path to outputdir to save model')
    parser_train.add_argument('--vocab',
                              help='absolute path to the the vocabulary file.'
                                   'Where to load and/or save the vocabulary')
    args = parser.parse_args()
    args.func(args)
