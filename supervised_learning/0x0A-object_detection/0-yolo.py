#!/usr/bin/env python3
""" Function Yolo Class Constructor
Write a class Yolo that uses the Yolo v3 algorithm to perform
object detection:

model_path is the path to where a Darknet Keras model is stored
classes_path is the path to where the list of class names used for the Darknet
model, listed in order of index, can be found
class_t is a float representing the box score threshold for the initial
filtering step
nms_t is a float representing the IOU threshold for non-max suppression
anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2) containing
all of the anchor boxes:
outputs is the number of outputs (predictions) made by the Darknet model
anchor_boxes is the number of anchor boxes used for each prediction
2 => [anchor_box_width, anchor_box_height]
Public instance attributes:
model: the Darknet Keras model
class_names: a list of the class names for the model
class_t: the box score threshold for the initial filtering step
nms_t: the IOU threshold for non-max suppression
anchors: the anchor boxes
"""
import tensorflow.keras as K


class Yolo:
    """Class Yolo v3"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class Yolo"""
        class_names = []
        model = K.models.load_model(model_path)
        f = open(classes_path, "r")
        for x in f:
            x_strip = x.strip()
            class_names.append(x_strip)
        f.close()
        self.model = model
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
