# Word2Vec

[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][travis-image]][travis-url]
[![MIT License][license-image]][license-url]

This is a re-implementation of Word2Vec relying on Tensorflow
[Estimators](https://www.tensorflow.org/guide/estimators) and
[Datasets](https://www.tensorflow.org/guide/datasets_for_estimators).

Works with python >= 3.6 and Tensorflow v2.0.

## Install
via pip:
```shell
pip3 install tf-word2vec
```
or, after a git clone:
```shell
python3 setup.py install
```

## Get data
You can download a sample of the English Wikipedia here:
```shell
wget http://129.194.21.122/~kabbach/enwiki.20190120.sample10.0.balanced.txt.7z
```

## Train Word2Vec
```shell
w2v train \
  --data /absolute/path/to/enwiki.20190120.sample10.0.balanced.txt \
  --outputdir /absolute/path/to/word2vec/models \
  --alpha 0.025 \
  --neg 5 \
  --window 2 \
  --epochs 5 \
  --size 300 \
  --min-count 50 \
  --sample 1e-5 \
  --train-mode skipgram \
  --t-num-threads 20 \
  --p-num-threads 25 \
  --keep-checkpoint-max 3 \
  --batch 1 \
  --shuffling-buffer-size 10000 \
  --save-summary-steps 10000 \
  --save-checkpoints-steps 100000 \
  --log-step-count-steps 10000
```

[release-image]:https://img.shields.io/github/release/akb89/word2vec.svg?style=flat-square
[release-url]:https://github.com/akb89/word2vec/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/tf-word2vec.svg?style=flat-square
[pypi-url]:https://pypi.org/project/tf-word2vec/
[travis-image]:https://img.shields.io/travis/akb89/word2vec.svg?style=flat-square
[travis-url]:https://travis-ci.org/akb89/word2vec
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt
