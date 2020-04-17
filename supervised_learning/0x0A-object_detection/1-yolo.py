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
        return (1/(1 + np.exp(-x)))    

    def process_outputs(self, outputs, image_size):
        """Function process output
        def decode_netout(netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
        for i in range(len(yolos)):
            boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
        """
        boxes = []
        box_confidence = []
        box_class_probs = []
        net_h = image_size[0]
        net_w = image_size[1]
        obj_tresh = 0.5
        nb_box = 3
        nb_class = 80
        for ii in range(len(outputs)):
            netout = outputs[ii]
            anchors = self.anchors[ii]
            grid_h, grid_w = netout.shape[:2]
            netout = netout.reshape((grid_h, grid_w, nb_box, -1))
            netout[..., :1]  = self.sigmoid(netout[..., :1])
            netout[..., 4:]  = self.sigmoid(netout[..., 4:])
            for i in range(grid_h):
                for j in range(grid_w):
                    row = i
                    col = j
                    for b in range(nb_box):
                        confidence = netout[int(row)][int(col)][b][4]
                        x, y, w, h = netout[int(row)][int(col)][b][:4]
                        x = (col + x) / grid_w # center position, unit: image width
                        y = (row + y) / grid_h # center position, unit: image height
                        w = anchors[b][0] * np.exp(w) / net_w # unit: image width
                        h = anchors[b][1] * np.exp(h) / net_h # unit: image height  
                        # last elements are class probabilities
                        classes = netout[int(row)][col][b][5:]
                        x1 = x-w/2
                        x2 = x+w/2
                        y1 = y-h/2
                        y2 = y+h/2
                        print("x1:", x1)
                        print("x2:", x2)
                        print("y1:", y1)
                        print("y2:", y2)
            np_bx = np.ndarray((grid_h, grid_w, nb_box, 4))
            np_bx = netout[...,:4]
            np_bx[...,:4] = x1, x2, y1, y2
            boxes.append(np_bx)
            np_bx_conf = np.ndarray((grid_h, grid_w, nb_box, 1))
            np_bx_conf = netout[...,4]
            np_bx_conf[...,0]=confidence
            box_confidence.append(np_bx_conf)
            np_bx_clss = np.ndarray((grid_h, grid_w, nb_box, nb_class)) 
            np_bx_clss = netout[...,5:]
            np_bx_clss[...,:] = classes
            box_class_probs.append(np_bx_clss)
        return (boxes, box_confidence, box_class_probs)

