#!/usr/bin/env python3
""" Function Conv Backward
Write a function that performs back propagation over a convolutional
layer of a neural network:

dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
partial derivatives with respect to the unactivated output of the convolutional
layer
m is the number of examples
h_new is the height of the output
w_new is the width of the output
c_new is the number of channels in the output
A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
output of the previous layer
h_prev is the height of the previous layer
w_prev is the width of the previous layer
c_prev is the number of channels in the previous layer
W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels
for the convolution
kh is the filter height
kw is the filter width
b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to
the convolution
padding is a string that is either same or valid, indicating the type of
padding used
stride is a tuple of (sh, sw) containing the strides for the convolution
sh is the stride for the height
sw is the stride for the width
you may import numpy as np
Returns: the partial derivatives with respect to the previous layer (dA_prev),
the kernels (dW), and the biases (db), respectively
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Function Convolution Forward"""
    """
    A naive implementation of the backward pass for a convolutional layer.
    Inputs:
    - dZ: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    
    dx, dw, db = None, None, None

    # Récupération des variables
    pad = 0
    if padding is 'same':
        pad = W.shape[0]
    
    stride = 1
   
    # Initialisations
    dA = np.zeros_like(A_prev)
    dw = np.zeros_like(W)
    db = np.zeros_like(b)
    
    # Dimensions

    N, H, Wx, C = A_prev.shape
    HH, WW, F, _ = W.shape
    _, H_, W_, _ = dZ.shape
    # db - dZ (N, F, H', Wx')
    # On somme sur tous les éléments sauf les indices des filtres
    db = np.sum(dZ, axis=(0, 1, 2))
    
    # dw = xp * dy
    # 0-padding juste sur les deux dernières dimensions de x
    Ap = np.pad(A_prev, ((0,), (pad,), (pad,), (0,)), 'constant')
    
    # Version sans vectorisation
    for n in range(N):       # On parcourt toutes les images
        for f in range(F):   # On parcourt tous les filtres
            for i in range(H_): # indices du résultat
                for j in range(W_):
                    for k in range(HH): # indices du filtre
                        for l in range(WW):
                            for c in range(C): # profondeur
                                dw[i,j,f,c] += Ap[n, stride*i+k, stride*j+l, c] * dZ[n,k, l, f]

    # dx = dy_0 * w'
    # Valide seulement pour un stride = 1
    # 0-padding juste sur les deux dernières dimensions de dy = dZ (N, F, H', W')
    dZp = np.pad(dZ, ((0,), (HH-1, ), (WW-1,), (0,)), 'constant')

    # 0-padding juste sur les deux dernières dimensions de dx
    dAp = np.pad(dA, ((0,), (pad,), (pad,), (0, )), 'constant')

    # filtre inversé dimension (F, C, HH, WW)
    w_ = np.zeros_like(W)
    for i in range(HH):
        for j in range(WW):
            w_[i, j, :, :] = W[HH-i-1,WW-j-1, :, :]
    
    # Version sans vectorisation
    for n in range(N):       # On parcourt toutes les images
        for f in range(F):   # On parcourt tous les filtres
            for i in range(H+2*pad): # indices de l'entrée participant au résultat
                for j in range(Wx+2*pad):
                    for k in range(HH): # indices du filtre
                        for l in range(WW):
                            for c in range(C): # profondeur
                                dAp[n,i,j,c] += dZp[n,i+k, j+l,f] * w_[k, l,f,c]
    #Remove padding for dA
    dA = dAp[:,pad:-pad,pad:-pad, :]

    return dA, dw, db
