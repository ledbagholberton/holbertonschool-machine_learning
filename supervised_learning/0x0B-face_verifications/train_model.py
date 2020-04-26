"""class TrainModel that trains a model for face verification
using triplet loss

Create the class constructor def __init__(self, model_path, alpha):
model_path is the path to the base face verification embedding model
loads the model using with tf.keras.utils.CustomObjectScope({'tf': tf}):
saves this model as the public instance method base_model
alpha is the alpha to use for the triplet loss calculation
Creates a new model:
inputs: [A, P, N]
A is a numpy.ndarray containing the anchor images
P is a numpy.ndarray containing the positive images
N is a numpy.ndarray containing the negative images
outputs: the triplet losses of base_model
compiles the model with Adam optimization and no additional losses
save this model as the public instance method training_model
you can use from triplet_loss import TripletLoss
"""
import tensorflow.keras as K
import tensorflow as tf
import numpy as np
from triplet_loss import TripletLoss

class TrainModel():
    """Class TrainModel"""
    def __init__(self, model_path, alpha):
        """Class TrainModel init"""
        with CustomObjectScope({'tf': tf}):
            model = load_model(model_path)
            self.base_model = model.save(base_model.h5)
        loss_layer = TripletLoss(alpha=alpha,name='triplet_loss')([A, P, N])
        self.training_model = build_model(inputs=[A, P, N],outputs=loss_layer)
        optimizer = Adam(lr = 0.00006)
        self.training_model.compile(loss=None,optimizer=optimizer)
        return(self.training_model)

    def train(self, triplets, epochs=5, batch_size=32,
                validation_split=0.3, verbose=True):
        """Method train
        triplets is a list containing the inputs to self.training_model
        epochs is the number of epochs to train for
        batch_size is the batch size for training
        validation_split is the validation split for training
        verbose is a boolean that sets the verbosity mode
        Returns: the History output from the training"""
        trained_network = network.fit(triplets,
                                      nb_epoch=epochs,
                                      batch_size=batch_size,
                                      verbose=verbose,
                                      validation_split=validation_split)
        return(trained_network.History)
    
    def save(self, save_path):
        """Method train
        Create the public instance method that saves the base embedding model
        save_path is the path to save the model
        Returns: the saved model"""
        return(model.save(save_path))

    @staticmethod
    def f1_score(y_true, y_pred):
        """Method f1 score"""
        #f1 = 2tp / (2tp + fp + fn)
        f1 = True
        return(f1)
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Method accuracy"""
        #acc = tp + tn / p + n
        acc = True
        return(acc)

    def best_tau(self, images, identities, thresholds):
        """Calculation of best tau

        images
        identities
        thresholds
        Returns: (tau, f1, acc)
        tau
        f1
        acc"""
        return(True)
