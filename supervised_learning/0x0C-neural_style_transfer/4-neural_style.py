#!/usr/bin/env python3
"""class NST"""
import tensorflow as tf
import numpy as np
tf.enable_eager_execution()


class NST:
    """Class NST"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ Constructor
        style_image - the image used as a style reference,
        stored as a numpy.ndarray
        content_image - the image used as a content reference,
        stored as a numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost
        if style_image is not a np.ndarray with the shape (h, w, 3),
        raise a TypeError with the message
        'style_image must be a numpy.ndarray with shape (h, w, 3)'
        if content_image is not a np.ndarray with the shape (h, w, 3),
        raise a TypeError with the message
        'content_image must be a numpy.ndarray with shape (h, w, 3)'
        if alpha is not a non-negative number,
        raise a TypeError with the message
        'alpha must be a non-negative number'
        if beta is not a non-negative number,
        raise a TypeError with the message
        'beta must be a non-negative number'
        Sets Tensorflow to execute eagerly
        Sets the instance attributes:
        style_image - the preprocessed style image
        content_image - the preprocessed content image
        alpha - the weight for content cost
        beta - the weight for style cost"""
        if (not isinstance(style_image, np.ndarray)
                or len(style_image.shape) is not 3
                or style_image.shape[2] is not 3):
            m1 = 'style_image must be a numpy.ndarray with shape (h, w, 3)'
            raise TypeError(m1)
        if (not isinstance(content_image, np.ndarray)
                or len(content_image.shape) is not 3
                or content_image.shape[2] is not 3):
            m2 = 'content_image must be a numpy.ndarray with shape (h, w, 3)'
            raise TypeError(m2)
        if (alpha < 0):
            raise TypeError('alpha must be a non-negative number')
        if (beta < 0):
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()
        (self.gram_style_features,
         self.content_feature) = self.generate_features()

    @staticmethod
    def scale_image(image):
        """rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        image - a numpy.ndarray of shape (h, w, 3) containing the image to
        be scaled
        if image is not a np.ndarray with the shape (h, w, 3),
        raise a TypeError with the
        message 'image must be a numpy.ndarray with shape (h, w, 3)'
        The scaled image should be a tf.tensor with the shape
        (1, h_new, w_new, 3)
        where max(h_new, w_new) == 512 and min(h_new, w_new)
        is scaled proportionately
        The image should be resized using bicubic interpolation
        The image’s pixel values should be rescaled from the range
        [0, 255] to [0, 1]
        Public class attributes:
        style_layers = ['block1_conv1', 'block2_conv1',
                        'block3_conv1', 'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'
        Returns: the scaled image
        """
        if (not isinstance(image, np.ndarray)
                or len(image.shape) is not 3
                or image.shape[2] is not 3):
            m3 = 'image must be a numpy.ndarray with shape (h, w, 3)'
            raise TypeError(m3)
        max_dim = 512
        long = max(image.shape[0], image.shape[1])
        scale = max_dim/long
        new_h = round(image.shape[0] * scale)
        new_w = round(image.shape[1] * scale)
        image = np.expand_dims(image, axis=0)
        image = tf.image.resize_bicubic(image, (new_h, new_w))
        image = tf.clip_by_value(image / 255, 0, 1)
        return image

    def load_model(self):
        """ Method Load Model
        creates the model used to calculate cost
        the model should use the VGG19 Keras model as a base
        the model’s input should be the same as the VGG19 input
        the model’s output should be a list containing the outputs of the
        VGG19 layers listed in style_layers followed by content _layer
        saves the model in the instance attribute model"""
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                weights='imagenet')
        input = vgg.input
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                input = tf.keras.layers.\
                        AveragePooling2D(name=layer.name)(input)
            else:
                input = layer(input)
        outputs = [vgg.get_layer(name).output
                   for name in self.style_layers]
        outputs = outputs + [vgg.get_layer(self.content_layer).output]
        global model
        model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
        model.trainable = False
        return(model)

    @staticmethod
    def gram_matrix(input_layer):
        """ Method gram matrix
        input_layer - an instance of tf.Tensor or tf.Variable of shape (1, h, w, c)
        containing the layer output whose gram matrix should be calculated
        if input_layer is not an instance of tf.Tensor or tf.Variable of rank 4,
        raise a TypeError with the message input_layer must be a tensor of rank 4
        Returns: a tf.Tensor of shape (1, c, c) containing the gram matrix of
        input_layer"""
        if (not isinstance(input_layer, tf.Tensor) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")
        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.reshape(gram, shape=[1, -1, channels])
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """ extracts the features used to calculate neural style cost
        Sets the public instance attributes:
        gram_style_features - a list of gram matrices calculated from the
        style layer outputs of the style image
        content_feature - the content layer output of the content image
        Add:
        model - the Keras model used to calculate cost
        gram_style_features - a list of gram matrices calculated from the
        style layer outputs of the style image
        content_feature - the content layer output of the content image"""
        num_style_layers = len(self.style_layers)
        # Load our images in the model VGG. It is necessary to preprocess it
        content_image = self.content_image
        style_image = self.style_image
        # aplico el modelo para cada imagen
        x = tf.keras.applications.vgg19.preprocess_input(style_image*255)
        y = tf.keras.applications.vgg19.preprocess_input(content_image*255)
        model_outputs = self.model(x) + self.model(y)
        # Adiciono las salidas de los modelos, por que asi son dados
        gram_style_features = [self.gram_matrix(layer) for layer in
                               model_outputs[:num_style_layers]]
        content_feature = model_outputs[-1:]
        return(gram_style_features, content_feature)

    def layer_style_cost(self, style_output, gram_target):
        """Style cost
        Calculates the style cost for a single layer
        style_output - tf.Tensor of shape (1, h, w, c) containing the layer
        style output of the generated image
        gram_target - tf.Tensor of shape (1, c, c) the gram matrix of the
        target style output for that layer
        if style_output is not an instance of tf.Tensor or tf.Variable of
        rank 4,
        raise a TypeError with the message style_output must be a tensor of
        rank 4
        if gram_target is not an instance of tf.Tensor or tf.Variable with
        shape (1, c, c), raise a TypeError with
        the message gram_target must be a tensor of shape [1, {c}, {c}]
        where {c} is the number of channels in style_output
        Returns: the layer’s style cost"""
        print("1")
        """if (not isinstance(style_output, tf.Tensor) or
                not isinstance(style_output, tf.Variable) or
                len(style_output.shape) is not 4):
            raise TypeError("style_output must be a tensor of rank 4")"""
        channels = style_output.shape[-1]
        c_gram_0 = gram_target.shape[0]
        c_gram_1 = gram_target.shape[1]
        """if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                len(gram_target.shape) is not 3 or
                c_gram_0 != c_gram_1 or
                c_gram_0 != channels):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {c}, {c}]")"""
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))
