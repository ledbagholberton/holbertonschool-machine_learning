#!/usr/bin/env python3
"""
references is a list of reference translations
each reference translation is a list of the words in the translation
sentence is a list containing the model proposed sentence
Returns: the unigram BLEU score
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    """
    row_sen = len(sentence)
    row_ref = len(references)
    col_ref = len(references[0])
    close = col_ref
    # creating a dict with key (the words in sentence) and the values are
    # the count of times each word in the references and the last file in
    # the list is the count of the word in the sentence.
    sen_dict = {word: [0]*(row_ref+1) for word in sentence}
    for key in sen_dict:
        for iter_ref in range(row_ref):
            col_ref = len(references[iter_ref])
            if abs(row_sen - close) > abs(col_ref - close):
                close = len(references[iter_ref])
            for iter_col in range(col_ref):
                if (key == references[iter_ref][iter_col]):
                    sen_dict[key][iter_ref] += 1
    for key in sen_dict:
        for iter_sen in sentence:
            if key == iter_sen:
                sen_dict[key][row_ref] += 1
    if row_sen <= col_ref:
        BP = np.exp((1 - (close/row_sen)))
    else:
        BP = 1
    values = np.array(tuple(sen_dict.values()))
    num = np.sum(np.max(values[:, 0:2], axis=1))
    den = np.sum(values[:, -1])
    pn = num / den
    bleu = BP * np.exp(np.log(pn))
    return bleu
