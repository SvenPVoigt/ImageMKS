import numpy as np
from math import floor, ceil, cos, sin
import scipy.ndimage as ndi

from .grids import divergent
from .shapes import circle
from ..common.size_check import circular_check


def gauss(sigma, theta=0, size=None, centered=True):
    '''
    Makes a circle with specified dtype. If bool or int, can be used as a mask.

    Parameters
    ----------
    r : The radius of the circle.
    size : The size of the output array that contains the object. Defaults to
        (round(2*r+1), round(2*r+1)).
    centered : If true, the center will be in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the center will be
        at the origin pixel (0,0). Defaults to True.
    '''
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


def conical(r, slope=1, size=None, centered=True):
    size = circular_check(r, size)

    if centered:
        c = [i//2 for i in size]
    else:
        c = [0,0]

    K = circle(r, size, centered)

    shifts = np.argwhere(K) - c
    cone = np.zeros(size)

    for i, s in enumerate(shifts):
        cone[c[0]+s[0], c[1]+s[1]] = slope * (1 - (np.sqrt(s[0]**2+s[1]**2)/r))

    cone[cone<0] = 0

    return cone


def drop(r, threshold=None, size=None, centered=True):
    size = circular_check(r, size)

    if centered:
        c = [i//2 for i in size]
    else:
        c = [0,0]

    K = circle(r, size, centered)
    K[c[0], c[1]] = 0

    shifts = np.argwhere(K) - c
    drop = np.zeros(size)
    for i, s in enumerate(shifts):
        drop[c[0]+s[0], c[1]+s[1]] = 1 / np.sqrt(s[0]**2+s[1]**2)

    drop[c[0], c[1]] = 1

    if threshold:
        drop[cone<threshold] = threshold

    return drop
