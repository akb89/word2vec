# Nonce2Vec

[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][travis-image]][travis-url]
[![MIT License][license-image]][license-url]

Welcome to Nonce2Vec!

**NEW** Nonce2Vec v3 went through a complete refactoring. Its architecture
is now designed around Tensorflow Estimators and Datasets.

## Install
```shell
pip3 install nonce2vec@3
```

## Train Word2Vec
You can train a Tensorflow implementation of Word2Vec via Non2Vec:

```shell
n2v3 train \
  --data /home/kabbach/nonce2vec/data/enwiki.opt.txt \
  --outputdir /home/kabbach/nonce2vec/models \
  --alpha 0.025 \
  --neg 5 \
  --window 15 \
  --epochs 5 \
  --size 400 \
  --min-count 50 \
  --sample 1e-5 \
  --train-mode cbow \
  --t-num-threads 20 \
  --p-num-threads 25 \
  --keep-checkpoint-max 3 \
  --batch 1 \
  --shuffling-buffer-size 10000 \
  --save-summary-steps 10000 \
  --save-checkpoints-steps 100000 \
  --log-step-count-steps 10000
```

[release-image]:https://img.shields.io/github/release/minimalparts/nonce2vec.svg?style=flat-square
[release-url]:https://github.com/minimalparts/nonce2vec/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/nonce2vec.svg?style=flat-square
[pypi-url]:https://pypi.org/project/nonce2vec/
[travis-image]:https://img.shields.io/travis/minimalparts/nonce2vec.svg?style=flat-square
[travis-url]:https://travis-ci.org/minimalparts/nonce2vec
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt
