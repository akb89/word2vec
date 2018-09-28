"""A word2vec implementation from scratch using Tensorflow."""

import tensorflow as tf


class Word2Vec():

    def __init__(self):
        self._vocabulary = []

    def build_vocab(self, input_filepath):
        with open(input_filepath, 'r') as input_stream:
            self._vocabulary = tf.compat.as_str(input_stream.read()).split()

    def train():
        pass


if __name__ == '__main__':
    INPUT_FILE = '/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.all.utf8.sent.split.lower'
    with open(INPUT_FILE, 'r') as input_stream:
        input_stream.read()
    # w2v = Word2Vec()
    # w2v.build_vocab(INPUT_FILE)
