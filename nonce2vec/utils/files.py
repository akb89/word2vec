"""Files utils."""

import os

__all__ = ('get_vocab_filepath', 'get_batches_filepath', 'get_model_dirpath')


def get_vocab_filepath(data_filepath, model_dirpath):
    """Return the absolute path to the vocabulary file."""
    os.makedirs(model_dirpath, exist_ok=True)
    return os.path.join(model_dirpath,
                        '{}.vocab'.format(os.path.basename(data_filepath)))


def get_batches_filepath(data_filepath, batch_size, model_dirpath):
    """Return the absolute path to the batches file."""
    os.makedirs(model_dirpath, exist_ok=True)
    return os.path.join(model_dirpath, '{}.batchsize{}.batches'.format(
        os.path.basename(data_filepath), batch_size))


def get_model_dirpath(datadir, outputdir, train_mode, alpha, neg, window_size,
                      sample, epochs, min_count, size, batch):
    """Return absolute path to w2v model directory."""
    model_dirpath = os.path.join(
        outputdir,
        '{}.{}.alpha{}.neg{}.win{}.sample{}.epochs{}.mincount{}.size{}.batch{}'
        .format(os.path.basename(datadir), train_mode, alpha, neg, window_size,
                sample, epochs, min_count, size, batch))
    os.makedirs(model_dirpath, exist_ok=True)
    return model_dirpath
