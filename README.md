# nonce2vec
This is the repo accompanying the paper "High-risk learning: acquiring new word vectors from tiny data" (Herbelot &amp; Baroni, 2017). 

**Abstract**

Distributional semantics models are known to struggle with small data. It is generally accepted that in order to learn 'a good vector' for a word, a model must have sufficient examples of its usage. This contradicts the fact that humans can guess the meaning of a word from a few occurrences only. In this paper, we show that a neural language model such as Word2Vec only necessitates minor modifications to its standard architecture to learn new terms from tiny data, using background knowledge from a previously learnt semantic space. We test our model on word definitions and on a nonce task involving 2-6 sentences' worth of context, showing a large increase in performance over state-of-the-art models on the definitional task. 

# Pre-requisites
You will need a pre-trained gensim model. You can go and train one yourself, using the gensim repo at [https://github.com/rare-technologies/gensim](https://github.com/rare-technologies/gensim), or simply download ours, pre-trained on Wikipedia: 

`wget http://clic.cimec.unitn.it/~aurelie.herbelot/wiki_all.model.tar.gz`

If you use our tar file, the content should be unpacked into the models/ directory of the repo.

# Running the code

Here is an example of how to run the code on the test set of the definitional dataset, with the best identified parameters from the paper:

`python test_def_nonces.py models/wiki_all.sent.split.model data/definitions/nonce.definitions.300.test 1 10000 3 15 1 70 1.9 5`

