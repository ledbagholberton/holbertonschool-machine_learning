#!/usr/bin/env python3

bag_of_words = __import__('0-bag_of_words').bag_of_words

sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The cake was not very good",
             "Life is beautiful"]
vocab = ["children", "is", "awesome", "cake", "are", "our", "future"]
E, F = bag_of_words(sentences, vocab)
print(E)
print(F)
