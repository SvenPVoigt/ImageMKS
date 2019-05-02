import numpy as np
from math import floor, ceil, cos, sin

from .common import circular_check
from .grids import divergent
from .shapes import circle


def gauss(sigma, theta=0, size=None, centered=True):
    '''
    Generates a 2d gaussian distribution. Usually used as a filter or kernel.

    Parameters
    ----------
    sigma : The standard deviation of the gaussian. First value of tuple is the
        x st. dev., second value is the y st. dev.. If one value is given,
        x and y st. dev. are the same.
    size : The size of the output array that contains the object. Defaults to
        (round(4*sigma+1), round(4*sigma+1)).
    centered : If true, the center will be in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the center will be
        at the origin pixel (0,0). Defaults to True.

    Returns
    -------
    A 2d gaussian distribution.
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
    '''
    Generates a 2d array of z values for a cone in range [0,1]. The location of each value is
    tied to the x,y location of the pixel.

    Parameters
    ----------
    r : The radius of the base of the cone
    slope : The slope of the sides of the cone. Determines how quickly the cone approaches the
        maximum value of 1.
    size : The size of the output array that contains the object. Defaults to
        (round(4*sigma+1), round(4*sigma+1)).
    centered : If true, the center will be in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the center will be
        at the origin pixel (0,0). Defaults to True.

    Returns
    -------
    A cone described by a 2d array.
    '''

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
    '''
    Generates a 2d array of z values for a drop shape in range [0,1]. The location of each
    z value is tied to the x,y location of the pixel.

    Parameters
    ----------
    r : The radius of the base of the drop.
    threshold : The relative values beneath which we determine the drop is equal to 0.
    size : The size of the output array that contains the object. Defaults to
        (round(4*sigma+1), round(4*sigma+1)).
    centered : If true, the center will be in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the center will be
        at the origin pixel (0,0). Defaults to True.

    Returns
    -------
    A drop described by a 2d array.
    '''

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
