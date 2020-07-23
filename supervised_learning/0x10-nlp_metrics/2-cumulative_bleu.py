#!/usr/bin/env python3
"""
references is a list of reference translations
each reference translation is a list of the words in the translation
sentence is a list containing the model proposed sentence
n is the size of the largest n-gram to use for evaluation
All n-gram scores should be weighted evenly
Returns: the cumulative n-gram BLEU score
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    calculates the n-gram Bcumulative BLEU score for a sentence
    """
    row_sen = len(sentence)
    row_ref = len(references)
    col_ref = len(references[0])
    close = col_ref
    # create the n-gram lists based on sentences and n
    list_gram = [[] for i in range(n+1)]
    # creates the n-grams for the sentence
    for n_iter in range(1, n+1):
        for i in range(row_sen - n_iter + 1):
            ngram = ''
            for j in range(n_iter):
                ngram = ngram + ' ' + sentence[i + j]
            list_gram[n_iter].append(ngram)
    # print("LIST_GRAM")
    # print(list_gram)
    # creates the n-grams in the references and stores it in new_list
    new_list = [[[] for i in range(n+1)] for j in range(row_ref)]
    for n_iter in range(1, n+1):
        # iterate on each reference. Create new_list
        for i in range(row_ref):
            col_ref = len(references[i])
            # iterates on each word in each reference
            for j in range(col_ref - n_iter + 1):
                new_word = ''
                # built the n-gram. Sum n_iter times the word with the nexts
                for k in range(n_iter):
                    new_word += ' ' + references[i][j + k]
                new_list[i][n_iter].append(new_word)
    # print("NEW LIST")
    # print(new_list)
    # generate a dictionary with all the n-grams from sentences
    new_dict = {}
    for n_iter in range(1, n + 1):
        new_dict[n_iter] = {word: [0]*(row_ref+1)
                            for word in list_gram[n_iter]}
    # print("NEW DICT ALONE")
    # print(new_dict)

    # compare n-grams in new_dict with n_grams from references
    for n_key in new_dict:
        for word_key in new_dict[n_key]:
            for ref in range(row_ref):
                size_list = len(new_list[ref][n_key])
                for n_gram in range(size_list):
                    if word_key == new_list[ref][n_key][n_gram]:
                        new_dict[n_key][word_key][ref] += 1
    # print("NEW DICT with COINCIDENCES IN REFERENCES")
    # print(new_dict)

    # Count n-grams in sentence (due to repeat n-grams)
    for n_key in new_dict:
        for word_key in new_dict[n_key]:
            size_list = len(list_gram[n_key])
            for n_gram in range(size_list):
                if word_key == list_gram[n_key][n_gram]:
                    new_dict[n_key][word_key][row_ref] += 1
    # print("NEW DICT with COINCIDENCES IN SENTENCES")
    # print(new_dict)

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

    pn = 0
    for n_key in new_dict:
        values = np.array(tuple(new_dict[n_key].values()))
        num = np.sum(np.max(values[:, 0:2], axis=1))
        den = np.sum(values[:, -1])
        pn = pn + (np.log(num/den)*(1/n))
    bleu = BP * np.exp(pn)
    return bleu
