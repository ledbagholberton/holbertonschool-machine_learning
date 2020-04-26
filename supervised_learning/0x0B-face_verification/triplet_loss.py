"""Class triplet_loss
Create a custom layer class TripletLoss that inherits from 
tensorflow.keras.layers.Layer:

alpha is the alpha value used to calculate the triplet loss
sets the public instance attribute alpha
"""
import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import cv2
import glob
import os


class TripletLoss(K.layers.Layer):
    """Class triplet_loss"""
    def __init__(self, alpha, **kwargs):
        """Class TripletLoss init"""
        self.alpha = alpha
        super(TripletLoss, self).__init__(**kwargs)



    def triplet_loss(self, inputs):
        """Create the public instance method
        inputs is a list containing the anchor, positive and
        negative output 
        tensors from the last layer of the model, respectively
        Returns: a tensor containing the triplet loss values"""
        anchor, positive, negative = inputs
        p_dist = K.backend.sum(K.backend.square(anchor-positive), axis=-1)
        n_dist = K.backend.sum(K.backend.square(anchor-negative), axis=-1)
        data_tf = K.backend.maximum(p_dist - n_dist + self.alpha, 0)
        return(data_tf)

    def call(self, inputs):
        """Create the public instance method def call(self, inputs):
        inputs is a list containing the anchor, positive, and negative output tensors
        from the last layer of the model, respectively
        adds the triplet loss to the graph
        Returns: the triplet loss tensor"""
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
    