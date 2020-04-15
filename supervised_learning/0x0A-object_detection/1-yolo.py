#!/usr/bin/env python3
""" Includes Method Process Outputs
outputs is a list of numpy.ndarrays containing the predictions
from the Darknet model for a single image:
output shape (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
grid_height & grid_width => the height and width of the grid used for
the output
anchor_boxes => the number of anchor boxes used
4 => (t_x, t_y, t_w, t_h)
1 => box_confidence
classes => class probabilities for all classes
image_size is a numpy.ndarray containing the image’s original
size [image_height, image_width]
Returns a tuple of (boxes, box_confidences, box_class_probs):
boxes: a list of numpy.ndarrays of shape 
(grid_height, grid_width, anchor_boxes, 4)
containing the processed boundary boxes for each output, respectively:
4 => (x1, y1, x2, y2)
(x1, y1, x2, y2) should represent the boundary box relative to original image
box_confidences: a list of numpy.ndarrays of shape (grid_height, grid_width,
anchor_boxes, 1) containing the box confidences for each output, respectively
box_class_probs: a list of numpy.ndarrays of shape (grid_height, grid_width,
anchor_boxes, classes) containing the box’s class probabilities for each output
"""
import tensorflow.keras as K
import numpy as np

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

    def process_outputs(self, outputs, image_size):
        """Function process output"""
        image_height = image_size[0]
        image_width = image_size[1]
        model_predict = K.predict(self)
        grid_height, grid_width, anchor_boxes, dim_k = model_predict.shape
        classes = dim_k - 4 -1 
        boxes = np.zeros((grid_height, grid_width, anchor_boxes, 4))
        boxes = model_predict[:,:,:,0:3]
        for i in range(grid_width):
            for j in range(grid_height):
                for k in range(anchor_boxes):
                    tx = model_predict[i,j,k,0]
                    ty = model_predict[i,j,k,1]
                    th = model_predict[i,j,k,2]
                    tw = model_predict[i,j,k,3]
                    x1 = ((image_width/grid_width)*i) + tx/2
                    y1 = ((image_height/grid_height)*i) + ty/2
                    x2 = x1 + tw
                    y2 = y1 + th
                    boxes[i,j,k,0] = model_predict[i,j,k,x1]
                    boxes[i,j,k,1] = model_predict[i,j,k,y1]
                    boxes[i,j,k,2] = model_predict[i,j,k,x2]
                    boxes[i,j,k,3] = model_predict[i,j,k,y2]
        box_confidences = np.zeros((grid_height, grid_width, anchor_boxes, 1))
        box_confidences = model_predict[:,:,:,4]
        box_class_probs = np.zeros((grid_height, grid_width, anchor_boxes, classes))
        box_class_probs = model_predict[:,:,:,5:]
        return((boxes, box_confidences, box_class_probs))
