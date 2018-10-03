"""A word2vec implementation from scratch using Tensorflow."""

import tensorflow as tf

from collections import defaultdict


class Word2Vec():

    def __init__(self):
        self._word_freq = defaultdict()
        self._idx2word = {}
        self._word2idx = {}

    def build_vocab(self, input_filepath):
        with open(input_filepath, 'r') as input_stream:
            for word in input_stream.split():
                self._word_freq += 1


    def train():
        pass


if __name__ == '__main__':
    INPUT_FILE = '/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.all.utf8.sent.split.lower'
    with open(INPUT_FILE, 'r') as input_stream:
        input_stream.read()
    # w2v = Word2Vec()
    # w2v.build_vocab(INPUT_FILE)
