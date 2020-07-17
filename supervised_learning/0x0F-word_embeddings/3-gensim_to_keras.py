#!/usr/bin/env python3
"""
Gets the converts the gensim word2vec model to a keras layer:

model is a trained gensim word2vec models
Returns: the trainable keras Embedding
"""
import gensim
import tensorflow.keras as K


def gensim_to_keras(model):
    """ creates and trains a gensim word2vec model
    """
    keras_model = K.models.Sequential()
    keras_model.add(model.wv.get_keras_embedding(train_embeddings=False))
    return (keras_model)
