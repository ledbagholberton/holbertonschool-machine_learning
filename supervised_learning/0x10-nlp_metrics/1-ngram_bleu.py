#!/usr/bin/env python3
"""
references is a list of reference translations
each reference translation is a list of the words in the translation
sentence is a list containing the model proposed sentence
n is the size of the n-gram to use for evaluation
Returns: the n-gram BLEU score
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    calculates the n-gram BLEU score for a sentence
    """
    row_sen = len(sentence)
    row_ref = len(references)
    col_ref = len(references[0])
    close = col_ref
    # create the n-gram lists based on sentences and n
    list_gram = [None] * (n+1)
    list_gram[n] = []
    # creates the n-grams for the sentence
    for i in range(row_sen - n + 1):
        ngram = ''
        for j in range(n):
            ngram = ngram + ' ' + sentence[i + j]
        list_gram[n].append(ngram)
    # creates the n-grams in the references and stores it in new_list
    new_list = [[] for y in range(row_ref)]
    # iterate on each reference
    for i in range(row_ref):
        col_ref = len(references[i])
        # iterates on each word in each reference
        for j in range(col_ref - 1):
            new_word = ''
            # built the n-gram. Sum n times the word with the nexts
            for k in range(n):
                new_word += ' ' + references[i][j + k]
            new_list[i].append(new_word)
    list_size = len(list_gram[n])
    new_dict = {word: [0]*(row_ref+1) for word in list_gram[n]}
    # print(new_dict)
    for key in new_dict:
        for iter in range(row_ref):
            for iter_ngram in range(len(new_list[iter])):
                if key == new_list[iter][iter_ngram]:
                    new_dict[key][iter] += 1
    for key in new_dict:
        for n_gram in list_gram[n]:
            if key == n_gram:
                new_dict[key][row_ref] += 1
    sen_dict = {word: [0]*(row_ref+1) for word in sentence}
    for key in sen_dict:
        for iter_ref in range(row_ref):
            col_ref = len(references[iter_ref])
            if abs(row_sen - close) > abs(col_ref - close):
                close = len(references[iter_ref])
    if row_sen <= col_ref:
        BP = np.exp((1 - (close/row_sen)))
    else:
        BP = 1
    values = np.array(tuple(new_dict.values()))
    num = np.sum(np.max(values[:, 0:2], axis=1))
    den = np.sum(values[:, -1])
    pn = num / den
    bleu = BP * np.exp(np.log(pn))
    return bleu
