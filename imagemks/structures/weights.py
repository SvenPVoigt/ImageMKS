import numpy as np
from math import floor, ceil, cos, sin
import scipy.ndimage as ndi

from .grids import divergent
from .shapes import circle


def gauss(sigma, theta=0, size=None, centered=True):
    if isinstance(sigma, list) or isinstance(sigma, tuple):
        sx, sy = sigma
    else:
        sx = sigma
        sy = sigma

    if size is None:
        r_x = 2*floor(sx)+1
        r_y = 2*floor(sy)+1

        coord = np.array([[-r_x, -r_y],
                          [-r_x, r_y],
                          [r_x, -r_y],
                          [r_x, r_y]
                         ])

        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]
                            ])

        coord = np.matmul(coord, rotation)
        size = (ceil(np.amax(coord[:,1]))-floor(np.amin(coord[:,1])), ceil(np.amax(coord[:,0]))-floor(np.amin(coord[:,0])))

    X, Y = divergent(size, centered)

    D = X**2 + Y**2

    a = cos(theta)**2 / (2 * sx**2) + sin(theta)**2 / (2 * sy**2)
    b = -sin(2*theta) / (4 * sx**2) + sin(2*theta) / (4 * sy**2)
    c = sin(theta)**2 / (2 * sx**2) + cos(theta)**2 / (2 * sy**2)

    return np.exp(-(a*X*X + 2*b*X*Y + c*Y*Y))


def dgauss():
    g = gaussian(s_x, s_y, theta_degrees, n)

    if phi_degrees is None:
        rads = 0
    else:
        rads = phi_degrees * np.pi / 180

    der_x = np.array([ [-1, 1], ])
    der_y = np.array([ [-1,], [1,] ])

    gx = ndi.convolve(g, der_x, mode='mirror')

    gy = ndi.convolve(g, der_y, mode='mirror')

    dg = np.cos(rads) * gx + np.sin(rads) * gy

    return (1/np.sum(np.abs(dg))) * dg


def d2gauss():
    pass


def radial():
    pass


def linear():
    pass


def parabolic_radial():
    pass


def parabolic_straight():
    pass


def epanechnikov():
    pass
