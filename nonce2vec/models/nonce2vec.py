"""Nonce2Vec model.

A modified version of gensim.Word2Vec.
"""

import copy
import logging
from collections import defaultdict

import numpy
from scipy.special import expit
from six import iteritems
from six.moves import xrange
from gensim.models.word2vec import Word2Vec, Word2VecVocab, Word2VecTrainables
from gensim.utils import keep_vocab_item
from gensim.models.keyedvectors import Vocab

__all__ = ('Nonce2Vec')

logger = logging.getLogger(__name__)


def train_sg_pair(model, word, context_index, alpha,
                  nonce_count, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None, compute_loss=False,
                  is_ft=False):
    if context_vectors is None:
        #context_vectors = model.wv.syn0
        context_vectors = model.wv.vectors
        print('cxt_vectors1 = {}'.format(len(model.wv.vectors)))

    if context_locks is None:
        #context_locks = model.syn0_lockf
        context_locks = model.trainables.vectors_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word (NN output)

    l1 = context_vectors[context_index]  # input word (NN input/projection layer)
    neu1e = numpy.zeros(l1.shape)

    # Only train the nonce
    if model.vocabulary.nonce is not None \
     and model.wv.index2word[context_index] == model.vocabulary.nonce \
     and word != model.vocabulary.nonce:
        lock_factor = context_locks[context_index]
        lambda_den = model.lambda_den
        exp_decay = -(nonce_count-1) / lambda_den
        if alpha * numpy.exp(exp_decay) > model.min_alpha:
            alpha = alpha * numpy.exp(exp_decay)
        else:
            alpha = model.min_alpha
        print('context_index = {}'.format(context_index))
        #print('l1 = {}'.format(l1))
        print('lock_factor = {}'.format(lock_factor))
        print('lambda_den = {}'.format(lambda_den))
        print('exp_decay = {}'.format(exp_decay))
        print('alpha = {}'.format(alpha))
        print('model.hs = {}'.format(model.hs))
        print('model.negative = {}'.format(model.negative))
        print('learn_hidden = {}'.format(learn_hidden))
        print('compute_loss = {}'.format(compute_loss))
        print('learn_vectors = {}'.format(learn_vectors))
        if model.negative:
            print('neg_labels = {}'.format(model.neg_labels))
            # use this word (label = 1) + `negative` other random words not
            # from this sentence (label = 0)
            word_indices = [predict_word.index]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(
                    model.random.randint(model.cum_table[-1]))
                print('w = {}'.format(w))
                if w != predict_word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            prod_term = numpy.dot(l1, l2b.T)
            fb = expit(prod_term)  # propagate hidden -> output
            gb = (model.neg_labels - fb) * alpha  # vector of error gradients
            # multiplied by the learning rate
            print('prod_term = {}'.format(prod_term))
            print('gb = {}'.format(gb))
            if learn_hidden:
                model.syn1neg[word_indices] += numpy.outer(gb, l1)
                # learn hidden -> output
            neu1e += numpy.dot(gb, l2b)  # save error

        if learn_vectors:
            l1 += neu1e * lock_factor  # learn input -> hidden
                # (mutates model.wv.syn0[word2.index], if that is l1)
    #print('neu1e = {}'.format(neu1e))
    return neu1e


def train_batch_sg(model, sentences, alpha, work=None, compute_loss=False):
    """
    Update skip-gram model by training on a sequence of sentences.
    Each sentence is a list of string tokens, which are looked up in
    the model's vocab dictionary. Called internally from `Word2Vec.train()`.
    This is the non-optimized, Python version. If you have cython
    installed, gensim will use the optimized version from word2vec_inner
    instead.
    """
    result = 0
    window = model.window
    for sentence in sentences:
        #print(model.random)
        #print(model.random.rand())
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in
                       model.wv.vocab and model.wv.vocab[w].sample_int
                       > model.random.rand() * 2 ** 32 or w == '___']
        # Count the number of times that we see the nonce
        nonce_count = 0
        for pos, word in enumerate(word_vocabs):
            # Note: we have got rid of the random window size
            start = max(0, pos - window)
            for pos2, word2 in enumerate(word_vocabs[start:(pos + window + 1)],
                                         start):
                # don't train on the `word` itself
                if pos2 != pos:
                    # If training context nonce, increment its count
                    if model.wv.index2word[word2.index] == \
                     model.vocabulary.nonce:
                        nonce_count += 1
                        print('word2.index = {}'.format(word2.index))
                        print('word = {}'.format(model.wv.index2word[word2.index]))
                        train_sg_pair(model,
                                      model.wv.index2word[word.index],
                                      word2.index, alpha, nonce_count,
                                      compute_loss=compute_loss)

        result += len(word_vocabs)
        if window - 1 >= 3:
            window = window - model.window_decay
        model.recompute_sample_ints()
        print('result = {}'.format(result))
    return result


class Nonce2VecVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3,
                 sorted_vocab=True, null_word=0):
        super(Nonce2VecVocab, self).__init__(max_vocab_size, min_count, sample,
                                             sorted_vocab, null_word)
        self.nonce = None

    def prepare_vocab(self, hs, negative, wv, update=False,
                      keep_raw_vocab=False, trim_rule=None,
                      min_count=None, sample=None, dry_run=False):
        """Apply vocabulary settings for `min_count`.
        (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).
        Calling with `dry_run=True` will only simulate the provided
        settings and report the size of the retained vocabulary,
        effective corpus length, and estimated memory requirements.
        Results are both printed via logging and returned as a dict.
        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.
        """
        print('trim_rule', trim_rule)
        print('id index2word', id(wv.index2word))
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        if not update:
            raise Exception('Nonce2Vec can only update a pre-existing vocabulary')
        logger.info('Updating model with new vocabulary')
        new_total = pre_exist_total = 0
        # New words and pre-existing words are two separate lists
        new_words = []
        pre_exist_words = []
        # If nonce is already in previous vocab, replace its label
        # (copy the original to a new slot, and delete original)
        if self.nonce is not None and self.nonce in wv.vocab:
            gold_nonce = '{}_true'.format(self.nonce)
            nonce_index = wv.vocab[self.nonce].index
            print('nonce_index = {}'.format(nonce_index))
            wv.vocab[gold_nonce] = wv.vocab[self.nonce]
            print('len index2word before = {}'.format(len(wv.index2word)))
            wv.index2word[nonce_index] = gold_nonce
            #del wv.index2word[wv.vocab[self.nonce].index]
            del wv.vocab[self.nonce]
            print('wv.index2word[nonce_index] = {}'.format(wv.index2word[nonce_index]))
            print('before wv.index2word = {}'.format(wv.index2word[len(wv.index2word) - 1]))
            print('raw_vocab = {}'.format(len(self.raw_vocab)))
            for word, v in iteritems(self.raw_vocab):
                # Update count of all words already in vocab
                if word in wv.vocab:
                    pre_exist_words.append(word)
                    pre_exist_total += v
                    if not dry_run:
                        wv.vocab[word].count += v
                else:
                    # For new words, keep the ones above the min count
                    # AND the nonce (regardless of count)
                    if keep_vocab_item(word, v, min_count,
                                       trim_rule=trim_rule) or word == self.nonce:
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            print('len index2word after = {}'.format(len(wv.index2word)))
                            wv.vocab[word] = Vocab(count=v,
                                                   index=len(wv.index2word))
                            wv.index2word.append(word)
                            print('after wv.index2word = {}'.format(wv.index2word[len(wv.index2word) - 1]))
                            print('len index2word last = {}'.format(len(wv.index2word)))
                    else:
                        drop_unique += 1
                        drop_total += v
            original_unique_total = len(pre_exist_words) \
                + len(new_words) + drop_unique
            pre_exist_unique_pct = len(pre_exist_words) \
                * 100 / max(original_unique_total, 1)
            new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
            logger.info('New added %i unique words (%i%% of original %i) '
                        'and increased the count of %i pre-existing words '
                        '(%i%% of original %i)', len(new_words),
                        new_unique_pct, original_unique_total,
                        len(pre_exist_words), pre_exist_unique_pct,
                        original_unique_total)
            retain_words = new_words + pre_exist_words
            retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        # Only retaining one subsampling notion from original gensim implementation
        else:
            threshold_count = sample * retain_total

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = wv.vocab[w].count
            word_probability = (numpy.sqrt(v / threshold_count) + 1) \
                * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info('deleting the raw counts dictionary of %i items',
                        len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info('sample=%g downsamples %i most-common words', sample,
                    downsample_unique)
        logger.info('downsampling leaves estimated %i word corpus '
                    '(%.1f%% of prior %i)', downsample_total,
                    downsample_total * 100.0 / max(retain_total, 1),
                    retain_total)

        # return from each step: words-affected, resulting-corpus-size,
        # extra memory estimates
        report_values = {
            'drop_unique': drop_unique, 'retain_total': retain_total,
            'downsample_unique': downsample_unique,
            'downsample_total': int(downsample_total),
            'num_retained_words': len(retain_words)
        }

        if self.null_word:
            print('NULL WORD')
            # create null pseudo-word for padding when using concatenative
            # L1 (run-of-words)
            # this word is only ever input – never predicted – so count,
            # huffman-point, etc doesn't matter
            self.add_null_word(wv)

        if self.sorted_vocab and not update:
            self.sort_vocab(wv)
        if hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree(wv)
        if negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table(wv)

        return report_values, pre_exist_words


class Nonce2VecTrainables(Word2VecTrainables):

    def __init__(self, vector_size=100, seed=1, hashfxn=hash):
        super(Nonce2VecTrainables, self).__init__(vector_size, seed, hashfxn)

    def prepare_weights(self, pre_exist_words, hs, negative, wv, model_random,
                        update=False):
        """Build tables and model weights based on final vocabulary settings."""
        # set initial input/projection and hidden weights
        if not update:
            raise Exception('prepare_weight on Nonce2VecTrainables should '
                            'always be used with update=True')
        else:
            self.update_weights(pre_exist_words, hs, negative, wv, model_random)

    def update_weights(self, pre_exist_words, hs, negative, wv, model_random):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        logger.info('updating layer weights')
        gained_vocab = len(wv.vocab) - len(wv.vectors)
        # newvectors = empty((gained_vocab, wv.vector_size), dtype=REAL)
        newvectors = numpy.zeros((gained_vocab, wv.vector_size),
                                 dtype=numpy.float32)

        # randomize the remaining words
        for i in xrange(len(wv.vectors), len(wv.vocab)):
            # construct deterministic seed from word AND seed argument
            # newvectors[i - len(wv.vectors)] = self.seeded_vector(
            #     wv.index2word[i] + str(self.seed), wv.vector_size)
            # Initialise to sum (NOTE: subsample to try and get rid of
            # function words)
            for w in pre_exist_words:
                if wv.vocab[w].sample_int > model_random.rand() * 2 ** 32 \
                 or w == '___':
                    # Adding w to initialisation
                    #print(w)
                    #print(newvectors[i-len(wv.vectors)])
                    #print(wv.vocab[w].index)
                    #print(wv.vectors[wv.vocab[w].index])
                    newvectors[i-len(wv.vectors)] += wv.vectors[
                        wv.vocab[w].index]
                    #print('newvectors[]')
                    #print(newvectors[i-len(wv.vectors)])

        # Raise an error if an online update is run before initial training on
        # a corpus
        if not len(wv.vectors):
            raise RuntimeError('You cannot do an online vocabulary-update of a '
                               'model which has no prior vocabulary. First '
                               'build the vocabulary of your model with a '
                               'corpus before doing an online update.')

        wv.vectors = numpy.vstack([wv.vectors, newvectors])
        if negative:
            self.syn1neg = numpy.vstack([self.syn1neg,
                                         numpy.zeros((gained_vocab,
                                                      self.layer1_size),
                                                     dtype=numpy.float32)])
        wv.vectors_norm = None

        # do not suppress learning for already learned words
        self.vectors_lockf = numpy.ones(len(wv.vocab),
                                        dtype=numpy.float32)  # zeros suppress learning


class Nonce2Vec(Word2Vec):

    MAX_WORDS_IN_BATCH = 10000

    def __init__(self, sentences=None, size=100, alpha=0.025, window=5,
                 min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, sg=1, hs=0, negative=5,
                 cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH, compute_loss=False,
                 callbacks=(), max_final_vocab=None, window_decay=0,
                 sample_decay=1.0):
        super(Nonce2Vec, self).__init__(sentences, size, alpha, window,
                                        min_count, max_vocab_size, sample,
                                        seed, workers, min_alpha, sg, hs,
                                        negative, cbow_mean, hashfxn, iter,
                                        null_word, trim_rule, sorted_vocab,
                                        batch_words, compute_loss, callbacks)
        self.trainables = Nonce2VecTrainables(seed=seed, vector_size=size,
                                              hashfxn=hashfxn)
        self.lambda_den = 0.0
        self.sample_decay = float(sample_decay)
        self.window_decay = int(window_decay)

    @classmethod
    def load(cls, *args, **kwargs):
        w2vec_model = super(Nonce2Vec, cls).load(*args, **kwargs)
        n2vec_model = cls()
        for key, value in w2vec_model.__dict__.items():
            setattr(n2vec_model, key, value)
        return n2vec_model

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences.

        Return 2-tuple `(effective word count after ignoring unknown words
        and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work)
        else:
            raise Exception('Nonce2Vec does not support cbow mode')
        return tally, self._raw_word_count(sentences)

    def build_vocab(self, sentences, update=False, progress_per=10000,
                    keep_raw_vocab=False, trim_rule=None, **kwargs):
        """Build vocabulary from a sequence of sentences.

        (can be a once-only generator stream).
        Each sentence is a iterable of iterables (can simply be a list of
        unicode strings too).
        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens,
                but for larger corpora,
            consider an iterable that streams the sentences directly from
                disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`,
                :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence`
                in :mod:`~gensim.models.word2vec` module for such examples.
        update : bool
            If true, the new words in `sentences` will be added to model's
                vocab.
        progress_per : int
            Indicates how many words to process before showing/updating the
                progress.
        """
        total_words, corpus_count = self.vocabulary.scan_vocab(
            sentences, progress_per=progress_per, trim_rule=trim_rule)
        self.corpus_count = corpus_count
        report_values, pre_exist_words = self.vocabulary.prepare_vocab(
            self.hs, self.negative, self.wv, update=update,
            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)
        report_values['memory'] = self.estimate_memory(
            vocab_size=report_values['num_retained_words'])
        self.trainables.prepare_weights(pre_exist_words, self.hs,
                                        self.negative, self.wv, self.random,
                                        update=update)

    def recompute_sample_ints(self):
        for w, o in self.wv.vocab.items():
            o.sample_int = int(round(float(o.sample_int) / float(self.sample_decay)))