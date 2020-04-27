"""Class Verification"""
from train_model import TrainModel

class FaceVerification:
    """Class FaceVerification"""
    def __init__(self, model, database, identities):
        """COnstructor
        model is either the fave verification embedding model or
        the path to where the model is stored
        database is a numpy.ndarray of all the face embeddings in the database
        identities is a list of identities corresponding to
        the embeddings in the database
        Sets the public instance attributes database and identities
        """
        with K.utils.CustomObjectScope({'tf': tf}):
            self.model = load_model(model)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        """ Method embedding 
        images are the images to retrieve the embeddings of
        Returns: a numpy.ndarray of embeddings"""
        embedding = self.model.predict(images)
        return(embedding)
        
    def verify(self, image, tau=0.5):
        """ Method verify
        image is the aligned image of the face to be verify
        tau is the maximum euclidean distance used for verification
        Returns: (identity, distance), or (None, None) on failure
        identity is the identity of the verified face
        distance is the euclidean distance between the verified face embedding
        and the identified database embedding"""


        identity = 
        distance = np.sum(np.square(emb1 - emb2))
        if distance > tau:
            return(identity, distance)
        else:
            return(None, None)
