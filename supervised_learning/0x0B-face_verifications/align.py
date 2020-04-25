#!/usr/bin/env python3
""" Function align
class constructor def __init__(self, shape_predictor_path):
shape_predictor_path is the path to the dlib shape predictor model
Sets the public instance attributes:
detector - contains dlib‘s default face detector
shape_predictor - contains the dlib.shape_predictor"""
import dlib
import numpy as np
import cv2


class FaceAlign:
    """Class FaceAlign"""
    def __init__(self, shape_predictor_path):
        """Method initial"""
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(shape_predictor_path)
        self.detector = detector
        self.shape_predictor = sp

    def detect(self, image):
        """method detect
        Detects a face in an image:
        image is a numpy.ndarray of rank 3 containing an image from
        which to detect a face
        Returns: a dlib.rectangle containing the boundary box for the
        face in the image, or None on failure
        If multiple faces are detected, return the dlib.rectangle with
        the largest area
        If no faces are detected, return a dlib.rectangle that is the
        same as the image"""
        dets = self.detector(image, 1)
        num_faces = len(dets)
        if num_faces is 0:
            return(None)
        elif num_faces is 1:
            return(dets[0])
        else:
            area, area_1 = 0, 0
            i_1 = 0
            for i, d in enumerate(dets):
                width = d.right() - d.left()
                height = d.bottom() - d.top()
                area = width * height
                if area > area_1:
                    area_1 = area
                    i_1 = i
            return(dets[i_1])

    def find_landmarks(self, image, detection):
        """ method facial landmarks
        image is a numpy.ndarray of an image from which to find facial
        landmarks 
        detection is a dlib.rectangle containing the boundary box of
        the face in the image
        Returns: a numpy.ndarray of shape (p, 2)containing the landmark points,
        or None on failure
        p is the number of landmark points
        2 is the x and y coordinates of the point"""
        shape = self.shape_predictor(image, detection)
        coords = np.zeros((68, 2))
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def align(self, image, landmark_indices, anchor_points, size=96):
        """ method  that aligns an image for face verification
        image is a numpy.ndarray of rank 3 containing the image to be aligned
        landmark_indices is a numpy.ndarray of shape (3,) containing the indices
        of the three landmark points that should be used for the affine transformation
        anchor_points is a numpy.ndarray of shape (3, 2) containing the destination
        points for the affine transformation, scaled to the range [0, 1]
        size is the desired size of the aligned image
        Returns: a numpy.ndarray of shape (size, size, 3) containing the aligned image,
        or None if no face is detected"""
        box = self.detect(image)
        if box is not None:
            landmarks = self.find_landmarks(image, box)
            src_points = landmarks[landmark_indices]
            dst_points = (anchor_points*size)
            warp_mat = cv2.getAffineTransform(src_points.astype('float32'), dst_points.astype('float32'))
            warp_dst = cv2.warpAffine(image, warp_mat, (size, size))
            return (warp_dst)
        else:
            return (None)
