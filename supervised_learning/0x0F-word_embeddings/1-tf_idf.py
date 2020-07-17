#!/usr/bin/env python3
"""
sentences is a list of sentences to analyze
vocab is a list of the vocabulary words to use for the analysis
If None, all words within sentences should be used
Returns: embeddings, features
embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
s is the number of sentences in sentences
f is the number of features analyzed
features is a list of the features used for embeddings
sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ creates a TF-IDF embedding
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = vectorizer.get_feature_names()
    vocab = X.toarray()
    return (vocab, embeddings)
