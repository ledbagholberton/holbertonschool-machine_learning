#!/usr/bin/env python3
"""File utils"""
import glob
import cv2
import os
import numpy as np


def load_images(images_path, as_array=True):
    """FUnction load_images
    images_path is the path to a directory from which to load images
    as_array is a boolean indicating whether the images should be loaded
    as one numpy.ndarray
    If True, the images should be a numpy.ndarray of shape (m, h, w, c)
    m is the number of images
    h, w, and c are the height, width, and number of channels of all images
    If False, the images should be as a list of individual numpy.ndarrays
    All images should be loaded in RGB format
    The images should be loaded in alphabetical order by filename
    Returns: images, filenames
    images is either a list/numpy.ndarray of all images
    filenames is a list of the filenames associated with each 
    image in images"""
    os.chdir(images_path)
    filenames = glob.glob('*')
    filenames.sort()
    os.chdir('../')
    images = []
    if as_array is True:
        images = cv2.imread(images_path + '/' + filenames[0])
        images = np.expand_dims(images, axis=0)
    else:
        images.append(cv2.imread(images_path + '/' + filenames[0]))
    for img in range(1, len(filenames)):
        load_image = cv2.imread(images_path + '/' + filenames[img])
        if as_array is True:
            load_image = np.expand_dims(load_image, axis=0)
            images = np.concatenate((images, load_image), axis=0)
        else:
            images.append(load_image)
    return(images, filenames)

def load_csv(csv_path, params={}):
    """ Also in utils.py, write a function  that loads the contents of a csv 
    file as a list of lists:
    csv_path is the path to the csv to load
    params are the parameters to load the csv with
    Returns: a list of lists representing the contents found in csv_path"""
    import csv


    fields = []
    rows = []
    csv.register_dialect('myDialect', params)
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, dialect='myDialect')
        rows = list(csvreader)
    return(rows)

def save_images(path, images, filenames):
    """path is the path to the directory in which the images 
    should be saved.
    images is a list/numpy.ndarray of images to save
    filenames is a list of filenames of the images to save
    Returns: True on success and False on failure"""
    for i in range(len(images)):
        full_name = path + '/' + filenames[i]
        try:
            cv2.imwrite(full_name, images[i])
        except Exception:
            pass
    return True 

def generate_triplets(images, filenames, triplet_names):
    """images is a numpy.ndarray of shape (n, h, w, 3) containing the various
    images in the dataset
    filenames is a list of length n containing the corresponding filenames
    for images
    triplet_names is a list of lists where each sublist contains the filenames
    of an anchor, positive, and negative image, respectively
    Returns: a list [A, P, N]
    A is a numpy.ndarray of shape (m, h, w, 3) containing the anchor images
    for all m triplets
    P is a numpy.ndarray of shape (m, h, w, 3) containing the positive images
    for all m triplets
    N is a numpy.ndarray of shape (m, h, w, 3) containing the negative images
    for all m triplets"""
    A_name = triplet_names[0][0]
    P_name = triplet_names[0][1]
    N_name = triplet_names[0][2]
    result_A = np.char.find(A_name + '.jpg', filenames)
    index_A = np.where(-1 != result_A)
    result_P = np.char.find(P_name + '.jpg', filenames)
    index_P = np.where(-1 != result_P)
    result_N = np.char.find(N_name + '.jpg', filenames)
    index_N = np.where(-1 != result_N)
    A = images[index_A[0], ...]
    P = images[index_P[0], ...]
    N = images[index_N[0], ...]
    for iter in range(1, len(triplet_names)):
        A_name = triplet_names[iter][0]
        P_name = triplet_names[iter][1]
        N_name = triplet_names[iter][2]
        result_A = np.char.find(A_name + '.jpg', filenames)
        index_A = np.where(-1 != result_A)
        result_P = np.char.find(P_name + '.jpg', filenames)
        index_P = np.where(-1 != result_P)
        result_N = np.char.find(N_name + '.jpg', filenames)
        index_N = np.where(-1 != result_N)
        array_A = images[index_A[0], ...]
        array_P = images[index_P[0], ...]
        array_N = images[index_N[0], ...]
        A = np.concatenate((A, array_A), axis=0)
        P = np.concatenate((P, array_P), axis=0)
        N = np.concatenate((N, array_N), axis=0)
    return(A, P, N)
