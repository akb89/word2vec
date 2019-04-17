#!/usr/bin/env python3
"""word2vec setup.py.

This file details modalities for packaging the word2vec application.
"""

from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tf-word2vec',
    description='Word2Vec implentation with Tensorflow Estimators and Datasets',
    author=' Alexandre Kabbach',
    author_email='akb@3azouz.net',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1.5',
    url='https://github.com/akb89/word2vec',
    download_url='https://github.com/akb89/word2vec/archive/0.1.0.tar.gz',
    license='MIT',
    keywords=['word2vec', 'word embeddings', 'tensorflow', 'estimators', 'datasets'],
    platforms=['any'],
    packages=['word2vec', 'word2vec.utils', 'word2vec.models',
              'word2vec.exceptions', 'word2vec.logging',
              'word2vec.estimators', 'word2vec.evaluation'],
    package_data={'word2vec': ['logging/*.yml', 'resources/*']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'w2v = word2vec.main:main'
        ],
    },
    test_suite='tests',
    install_requires=['pyyaml>=5.1', 'tensorflow>=1.13.1'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Environment :: Web Environment',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries :: Python Modules',
                 'Topic :: Text Processing :: Linguistic'],
    zip_safe=False,
)
