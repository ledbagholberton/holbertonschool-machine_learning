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
        """Function process output"""
        image_height = image_size[0]
        image_width = image_size[1]
        yolo13 = outputs[0]
        yolo26 = outputs[1]
        yolo52 = outputs[2]
        """Box confidences"""
        conf13 = np.apply_along_axis(self.sigmoid, axis=3, arr=yolo13[...,3:4])
        conf26 = np.apply_along_axis(self.sigmoid, axis=3, arr=yolo26[...,3:4])
        conf52 = np.apply_along_axis(self.sigmoid, axis=3, arr=yolo52[...,3:4])
        box_confidences = [conf13, conf26, conf52]
        """Box classes"""
        class13 = np.apply_along_axis(self.sigmoid, axis=3, arr=yolo13[:,:,:,5:85])
        class26 = np.apply_along_axis(self.sigmoid, axis=3, arr=yolo26[:,:,:,5:85])
        class52 = np.apply_along_axis(self.sigmoid, axis=3, arr=yolo52[:,:,:,5:85])
        box_class_probs = [class13, class26, class52]
        """Boxes"""
        box13 = yolo13[:,:,:,0:3]
        tx13 = box13[:,:,:,0]
        ty13 = box13[:,:,:,1]
        tw13 = box13[:,:,:,2]
        th13 = box13[:,:,:,3]
        for i in box13.shape[0]:
            for j in box13.shape[1]:
                for k in box13.shape[2]:
                    cx13 = (image_width/13)*j
                    cy13 = (image_height/13)*i
                    bx13 = self.sigmoid(tx13) + cx13
                    by13 = self.sigmoid(ty13) + cy13
                    bw13 = np.exp(tw13) * k
                    bh13 = np.exp(tw13) * k
        
        box26 = yolo26[:,:,:,0:3]
        box52 = yolo52[:,:,:,0:3]
        boxes = [box13, box26, box52]

        """All lists"""
        return((boxes, box_confidences, box_class_probs))


