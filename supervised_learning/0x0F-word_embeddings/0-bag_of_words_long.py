# -*- coding: utf-8 -*-
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
import numpy as np


def bag_of_words(sentences, vocab=None):
    """ Create a Bag of words embedding matrix
    """
    texts = []
    if vocab is None:
        vocab = []
        for document in sentences:
            for word in document.lower().split():
                if "'s" in word:
                    word = word.replace("s", "")
                b = "!'@#$?=&/()"
                for char in b:
                    word = word.replace(char,"")
                texts.append(word)
        for text in texts:
            if not (text in vocab):
                vocab.append(text)
        vocab.sort()
    if texts is None:
        for document in sentences:
            for word in document.lower().split():
                if "'s" in word:
                    word = word.replace("s", "")
                b = "!'@#$?=&/()"
                for char in b:
                    word = word.replace(char,"")
                texts.append(word)
    else:
        count = 0
        pos = 0
        num_sentence = 0
        f = len(vocab)
        s = len(sentences)
        embeddings = np.zeros((s, f)).astype(int)
        for document in sentences:
            count = len(document.split())
            for i in range(count):
                for j in range(len(vocab)):
                    if texts[pos + i] == vocab[j]:
                        embeddings[num_sentence][j] += 1
            pos = count + pos
            num_sentence += 1
    return (embeddings, vocab)
