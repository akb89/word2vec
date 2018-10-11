"""Files utils."""

import os

__all__ = ('get_vocab_filepath', 'get_batches_filepath', 'get_model_path')


def get_vocab_filepath(data_filepath, min_count, model_dirpath):
    """Return the absolute path to the vocabulary file."""
    os.makedirs(model_dirpath, exist_ok=True)
    return os.path.join(model_dirpath, '{}.mincount{}.vocab'.format(
        os.path.basename(data_filepath), min_count))


def get_batches_filepath(data_filepath, batch_size, model_dirpath):
    """Return the absolute path to the batches file."""
    os.makedirs(model_dirpath, exist_ok=True)
    return os.path.join(model_dirpath, '{}.batchsize{}.batches'.format(
        os.path.basename(data_filepath), batch_size))


def get_model_path(datadir, outputdir, train_mode, alpha, neg, window_size,
                   sample, epochs, min_count, size):
    """Return absolute path to w2v model file.

    Model absolute path is computed from the outputdir and the
    datadir name.
    """
    os.makedirs(outputdir, exist_ok=True)
    return os.path.join(
        outputdir,
        '{}.{}.alpha{}.neg{}.win{}.sample{}.epochs{}.mincount{}.size{}.model'
        .format(os.path.basename(datadir), train_mode, alpha, neg, window_size,
                sample, epochs, min_count, size))
