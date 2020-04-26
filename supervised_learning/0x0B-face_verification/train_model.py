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
        with K.utils.CustomObjectScope({'tf': tf}):
            model = load_model(model_path)
            self.base_model = model.save(base_model.h5)
        self.alpha = alpha
        A = K.Input(input_shape, name='A')
        P = K.Input(input_shape, name='P')
        N = K.Input(input_shape, name='N')
        A_encoded = self.base_model(A)
        P_encoded = self.base_model(P)
        N_encoded = self.base_model(N)
        in_encoded = [A_encoded, P_encoded, N_encoded]
        loss_layer = TripletLoss(alpha=alpha,name='triplet_loss')(in_encoded)
        self.training_model = build_model(inputs=encoded,outputs=loss_layer)
        self.training_model.compile(loss=None,optimizer='Adam')
        self.training_model.save()

    def train(self, triplets, epochs=5, batch_size=32,
                validation_split=0.3, verbose=True):
        """Method train
        triplets is a list containing the inputs to self.training_model
        epochs is the number of epochs to train for
        batch_size is the batch size for training
        validation_split is the validation split for training
        verbose is a boolean that sets the verbosity mode
        Returns: the History output from the training"""
        t_network = self.training_model.fit(triplets,
                                            nb_epoch=epochs,
                                            batch_size=batch_size,
                                            verbose=verbose,
                                            validation_split=validation_split)
        return(t_network.History)
    
    def save(self, save_path):
        """Method train
        Create the public instance method that saves the base embedding model
        save_path is the path to save the model
        Returns: the saved model"""
        return(model.save(save_path))

    @staticmethod
    def f1_score(y_true, y_pred):
        """Method f1 score"""
        # tp: true positives
        tp = K.backend.sum(K.backend.round(K.backend.clip(y_true * y_pred, 0, 1)))
        # p_p: possible positives 
        p_p = K.backend.sum(K.backend.round(K.backend.clip(y_true, 0, 1)))
        # prp: predicted positives
        prp = K.backend.sum(K.backend.round(K.backend.clip(y_pred, 0, 1)))
        precision = tp / (prp + K.backend.epsilon())
        recall = tp / (p_p + K.backend.epsilon())
        f1_val = 2*(precision * recall)/(precision + recall + K.backend.epsilon())
        return(f1_val)
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Method accuracy"""
        acc = K.mean(K.equal(y_true, K.round(y_pred)))
        return(acc)
