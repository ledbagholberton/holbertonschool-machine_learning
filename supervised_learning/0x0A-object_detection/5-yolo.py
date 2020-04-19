#!/usr/bin/env python3
""" Includes Method Filter_boxes
Add the public method
boxes:
a list of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, 4)
containing the processed boundary boxes for each output, respectively
box_confidences:
a list of np.ndarrays shape (grid_height, grid_width, anchor_boxes, 1)
containing the processed box confidences for each output, respectively
box_class_probs:
a list numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, classes)
containing the processed box class probabilities for each output, respectively
Returns a tuple of (filtered_boxes, box_classes, box_scores):
filtered_boxes: a numpy.ndarray of shape (?, 4)
containing all of the filtered bounding boxes:
box_classes: a numpy.ndarray of shape (?,)
containing the class number that each box in filtered_boxes predicts, respec
box_scores: a numpy.ndarray of shape (?)
containing the box scores for each box in filtered_boxes, respectively
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob


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
        input_h = self.model.input.shape[2].value
        input_w = self.model.input.shape[1].value
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
                        x, y, w, h = netout[j][i][b][:4]
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

    def process_outputs(self, outputs, image_size):
        """Function process output"""
        boxes = []
        box_confidence = []
        box_class_probs = []
        net_h = image_size[0]
        net_w = image_size[1]
        obj_tresh = 0.5
        nb_box = 3
        nb_class = 80
        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value
        for ii in range(len(outputs)):
            netout = outputs[ii]
            anchors = self.anchors[ii]
            grid_h, grid_w = netout.shape[:2]
            netout[..., :2] = self.sigmoid(netout[..., :2])
            netout[..., 4:] = self.sigmoid(netout[..., 4:])
            np_bx = netout[..., :4]
            for i in range(grid_w):
                for j in range(grid_h):
                    for b in range(nb_box):
                        x, y, w, h = netout[i][j][b][:4]
                        x = (x + i)
                        new_x = x * (net_w / grid_w)
                        y = (y + j)
                        new_y = y * (net_h / grid_h)
                        w = (anchors[b][0] * np.exp(w))
                        new_w = w * (net_w / input_w)
                        h = (anchors[b][1] * np.exp(h))
                        new_h = h * (net_h / input_h)
                        x1 = (new_x - new_w/2)
                        y1 = (new_y - new_h/2)
                        x2 = (new_x + new_w/2)
                        y2 = (new_y + new_h/2)
                        np_bx[i, j, b, 0:4] = x1, y1, x2, y2
            boxes.append(np_bx)
            np_bx_conf = netout[..., 4:5]
            box_confidence.append(np_bx_conf)
            np_bx_clss = netout[..., 5:]
            box_class_probs.append(np_bx_clss)
        return (boxes, box_confidence, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Function filter boxes"""
        box_init = []
        # multiplico en cada array de la lista el confidence por la Pr de clase
        for iter_a, iter_b in zip(box_confidences, box_class_probs):
            box_init.append(iter_a * iter_b)
        # saco los maximos de cada clase en cada array de la lista
        box_score_max = [np.max(var_1, axis=-1) for var_1 in box_init]
        # aplano cada uno de los  array de  la lista a una sola dimension
        box_score_flat = [var_2.reshape(-1) for var_2 in box_score_max]
        # concateno cada array de la lista a un solo array
        box_total = np.concatenate(box_score_flat, axis=-1)
        # encuentro las posiciones en donde la multiplicacion < class_t
        pos = np.where(box_total >= self.class_t)
        # calculo el array box_scores como el box_total pero solo
        # con las posiciones encontradas antes
        box_scores = box_total[pos]
        # similar para filtered boxes
        filtered_boxes_flat = [var_3.reshape(-1, 4) for var_3 in boxes]
        filtered_boxes_total = np.concatenate(filtered_boxes_flat, axis=0)
        filtered_boxes = filtered_boxes_total[pos]
        # similar para box_classes pero antes hallo el max entre las clases
        box_classes_max = [var_4.argmax(axis=-1) for var_4 in box_class_probs]
        box_classes_flat = [var_5.reshape(-1) for var_5 in box_classes_max]
        box_classes_total = np.concatenate(box_classes_flat, axis=-1)
        box_classes = box_classes_total[pos]
        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ Function non max supression
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
        filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
        for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
        each box in filtered_boxes, respectively
        Returns a tuple of (box_predictions, predicted_box_classes,
        predicted_box_scores):
        box_predictions: a numpy.ndarray of shape (?, 4) containing all of the
        predicted bounding boxes ordered by class and box score
        predicted_box_classes: a numpy.ndarray of shape (?,) containing class
        number for box_predictions ordered by class and box score, respectively
        predicted_box_scores: a numpy.ndarray of shape (?) containing the box
        scores for box_predictions ordered by class and box score, respectively
        """
        n_boxes, n_classes, n_score = [], [], []
        # itero en un conjunto ordenado por clases
        for c in set(box_classes):
            # encuentro los indices en donde esta el iterador clase
            indx = np.where(box_classes == c)
            # hallo los subconjuntos que tienen cada clase para cada uno de
            # arrays de entrada a la funcion.
            b = filtered_boxes[indx]
            c = box_classes[indx]
            s = box_scores[indx]
            # llamo a la funcion auxiliar nms_boxes que encuentra los nms para
            # cada clase
            keep = self.nms_boxes(b, s)
            # retorna el index de mejor caja luego de aplicar NonMaxSupression
            # aplico los index en los arrays dados
            n_boxes.append(b[keep])
            n_classes.append(c[keep])
            n_score.append(s[keep])
        box_predictions = np.concatenate(n_boxes)
        predicted_box_classes = np.concatenate(n_classes)
        predicted_box_scores = np.concatenate(n_score)
        return (box_predictions,
                predicted_box_classes,
                predicted_box_scores)

    def nms_boxes(self, b, s):
        """Functio auxiliar keep"""
        x1 = b[..., 0]
        y1 = b[..., 1]
        x2 = b[..., 2]
        y2 = b[..., 3]
        anchor_area = (x2 - x1) * (y2 - y1)
        # ordeno los box_scores que traen todos los max scores por cada caja
        # en la ultima dimension
        order = s.argsort()[::-1]

        my_list = []
        # ciclo de depuracion del array ordenado mientras halla elementos
        while order.size > 0:
            # tomo el mayor
            i = order[0]
            my_list.append(i)
            # encuentro los valores maximo entre la posicion 0 y el resto
            xx2 = np.maximum(x1[i], x1[order[1:]])
            yy2 = np.maximum(y2[i], y2[order[1:]])
            xx1 = np.minimum(x1[i], x1[order[1:]])
            yy1 = np.minimum(y1[i], y1[order[1:]])
            # halla el maximo ancho y alto
            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            # halla el area maxima
            inter = w1 * h1
            # encuentra el IoU
            ovr = inter / (anchor_area[i] + anchor_area[order[1:]] - inter)
            # halla los indices en donde el ovr sea mayor que el requerido
            inds = np.where(ovr <= self.nms_t)[0]
            # depura la lista ordenada por scores
            order = order[inds + 1]
        # convierte a un arreglo la lista donde estan los que
        # no fueron suprimidos
        keep = np.array(my_list)
        return (keep)

    @staticmethod
    def load_images(folder_path):
        """ folder_path: a string representing the path to the folder holding
            all the images to load
            Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images"""
        images = []
        image_paths = glob.glob(folder_path + '/*')
        for img in image_paths:
            image = cv2.imread(img, 1)
            images.append(image)
        return (images, image_paths)

    def preprocess_images(self, images):
        """ Function preprocess images
        images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
        pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3) containing
        all of the preprocessed images
        ni: the number of images that were preprocessed
        input_h: the input height for the Darknet model
        Note: this can vary by model
        input_w: the input width for the Darknet model
        Note: this can vary by model
        3: number of color channels
        image_shapes: a numpy.ndarray of shape (ni, 2) containing the original
        height and width of the images
        2 => (image_height, image_width)"""
        n_images = []
        shape = []
        ni = 0
        for img in images:
            image_height = self.model.input.shape[2].value
            image_width = self.model.input.shape[1].value
            image = cv2.resize(img, (image_height, image_width),
                               interpolation=cv2.INTER_CUBIC)
            image = image.astype(float) / 255
            n_images.append(image)
        pimages = np.stack(n_images, axis=0)
        shape = [img.shape[:2] for img in images]
        image_shapes = np.stack(shape, axis=0)
        return (pimages, image_shapes)
