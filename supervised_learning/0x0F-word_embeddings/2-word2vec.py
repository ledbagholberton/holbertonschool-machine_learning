#!/usr/bin/env python3
"""
sentences is a list of sentences to be trained on
size is the dimensionality of the embedding layer
min_count is the minimum number of occurrences of a word for use in training
window is the maximum distance between the current and
predicted word within a sentence
negative is the size of negative sampling
cbow is a boolean to determine the training type; True is for CBOW;
False is for Skip-gram
iterations is the number of iterations to train over
seed is the seed for the random number generator
Returns: the trained model
"""
import gensim


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0):
    """ creates and trains a gensim word2vec model
    """
    if cbow is True:
        sg = 0
    else:
        sg = 1
    model = gensim.models.Word2Vec(sentences=sentences, min_count=min_count,
                                   size=size, seed=seed, iter=iterations+1,
                                   window=window, negative=negative, sg=sg)
    return (model)
