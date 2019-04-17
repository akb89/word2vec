"""Welcome to word2vec.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

from word2vec.estimators.word2vec import Word2Vec

import word2vec.utils.config as cutils
import word2vec.utils.files as futils

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
        w2v.build_vocab(args.datafile, vocab_filepath, args.min_count)
    else:
        w2v.load_vocab(args.vocab, args.min_count)
    w2v.train(args.train_mode, args.datafile, output_model_dirpath,
              args.batch, args.size, args.neg, args.alpha,
              args.window, args.epochs, args.sample, args.p_num_threads,
              args.t_num_threads, args.shuffling_buffer_size,
              args.save_summary_steps,
              args.save_checkpoints_steps, args.keep_checkpoint_max,
              args.log_step_count_steps, args.debug, args.debug_port, args.xla)


def main():
    """Launch word2vec."""
    parser = argparse.ArgumentParser(prog='word2vec')
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser(
        'train', formatter_class=argparse.RawTextHelpFormatter,
        help='generate pre-trained embeddings from wikipedia dump via '
             'a Tensorflow implementation of Word2Vec')
    parser_train.set_defaults(func=_train)
    parser_train.add_argument('--p-num-threads', type=int, default=1,
                              help='number of threads used for preprocessing')
    parser_train.add_argument('--t-num-threads', type=int, default=1,
                              help='number of threads used for training')
    parser_train.add_argument('--alpha', type=float, default=0.025,
                              help='initial learning rate')
    parser_train.add_argument('--neg', type=int, default=5,
                              help='number of negative samples')
    parser_train.add_argument('--window', type=int, default=2,
                              help='window size')
    parser_train.add_argument('--sample', type=float, default=1e-5,
                              help='subsampling rate')
    parser_train.add_argument('--epochs', type=int, default=5,
                              help='number of epochs')
    parser_train.add_argument('--min-count', type=int, default=50,
                              help='min frequency count')
    parser_train.add_argument('--data', required=True, dest='datafile',
                              help='absolute path to training data file')
    parser_train.add_argument('--size', type=int, default=300,
                              help='vector dimensionality')
    parser_train.add_argument('--batch', type=int, default=1,
                              help='batch size')
    parser_train.add_argument('--shuffling-buffer-size', type=int,
                              default=10000, help='size of buffer to use for '
                                                  'shuffling training data')
    parser_train.add_argument('--save-summary-steps', type=int,
                              default=100000, help='save summaries every this '
                                                   'many steps')
    parser_train.add_argument('--save-checkpoints-steps', type=int,
                              default=1000000, help='save checkpoints every '
                                                    'this many steps')
    parser_train.add_argument('--keep-checkpoint-max', type=int,
                              default=3,
                              help='the maximum number of recent checkpoint '
                                   'files to keep. As new files are created, '
                                   'older files are deleted. If None or 0, all'
                                   ' checkpoint files are kept')
    parser_train.add_argument('--log-step-count-steps', type=int,
                              default=100000,
                              help='the frequency, in number of global steps, '
                                   'that the global step/sec and the loss will'
                                   ' be logged during training')
    parser_train.add_argument('--train-mode', choices=['cbow', 'skipgram'],
                              default='skipgram',
                              help='how to train word2vec')
    parser_train.add_argument('--outputdir', required=True,
                              help='absolute path to outputdir to save model')
    parser_train.add_argument('--vocab',
                              help='absolute path to the the vocabulary file.'
                                   'Where to load and/or save the vocabulary')
    parser_train.add_argument('--debug', action='store_true',
                              help='set up a debugger hook for Tensorboard')
    parser_train.add_argument('--debug-port', type=int, default=2333,
                              help='the debugger port, as specified in '
                                   'Tensorboard')
    parser_train.add_argument('--xla', action='store_true',
                              help='run Tensorflow with XLA JIT compilation')
    args = parser.parse_args()
    args.func(args)
