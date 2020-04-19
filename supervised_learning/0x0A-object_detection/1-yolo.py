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

    def sigmoid(self, x):
        """Simoid function """
        return (1/(1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """Function process output"""
        boxes = []
        box_confidence = []
        box_class_probs = []
        net_h = image_size[0]
        net_w = image_size[1]
        input_w = self.model.input.shape[2].value
        input_h = self.model.input.shape[1].value
        for ii in range(len(outputs)):
            netout = outputs[ii]
            nb_box = netout.shape[-2]
            nb_class = netout.shape[-1] - 5
            anchors = self.anchors[ii]
            grid_h, grid_w = netout.shape[:2]
            netout[..., :2] = self.sigmoid(netout[..., :2])
            netout[..., 4:] = self.sigmoid(netout[..., 4:])
            np_bx = netout[..., :4]
            for j in range(grid_h):
                for i in range(grid_w):
                    for b in range(nb_box):
                        x, y, w, h = netout[i][j][b][:4]
                        x = (x + i)
                        new_x = x / grid_w
                        y = (y + j)
                        new_y = y / grid_h
                        w = (anchors[b][0] * np.exp(w))
                        new_w = w / input_w
                        h = (anchors[b][1] * np.exp(h))
                        new_h = h / input_h
                        x1 = (new_x - new_w/2) * net_w
                        y1 = (new_y - new_h/2) * net_h
                        x2 = (new_x + new_w/2) * net_w
                        y2 = (new_y + new_h/2) * net_h
                        np_bx[j, i, b, 0:4] = x1, y1, x2, y2
            boxes.append(np_bx)
            np_bx_conf = netout[..., 4:5]
            box_confidence.append(np_bx_conf)
            np_bx_clss = netout[..., 5:]
            box_class_probs.append(np_bx_clss)
        return (boxes, box_confidence, box_class_probs)
