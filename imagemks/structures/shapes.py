import numpy as np
from math import floor, ceil
from .grids import divergent


def circle(r, size=None, centered=True, dtype=np.bool_):
    '''
    Makes a circle with specified dtype. If bool or int, can be used as a mask.

    Parameters
    ----------
    r : The radius of the circle.
    size : The size of the output array that contains the circle. Defaults to
        (round(2*r+1), round(2*r+1)).
    centered : If true, the circle will be centered in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the circle will be centered
        at the origin pixel (0,0). Defaults to True.
    dtype : A valid numpy dtype. Defaults to boolean.

    Returns
    -------
    A circle that obeys the equation :math:`x^2 + y^2 < r^2`
    '''

    if size is None:
        size = (round(2*r+1), round(2*r+1))

    X, Y = divergent(size, centered)

    return (X**2 + Y**2 <= r**2).astype(dtype)


def donut(r_outer, r_inner, size=None, centered=True, dtype=np.bool_):
    '''
    Makes a 2d donut with specified dtype. If bool or int, can be used as a mask.

    Parameters
    ----------
    r_outer : The radius of the outer border.
    r_inner : The radius of the inner border.
    size : The size of the output array that contains the donut. Defaults to
        (round(2*r_outer+1), round(2*r_outer+1)).
    centered : If true, the donut will be centered in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the donut will be centered
        at the origin pixel (0,0). Defaults to True.
    dtype : A valid numpy dtype. Defaults to boolean.

    Returns
    -------
    A donut that obeys the equation :math:`r_inner^2 < x^2 + y^2 < r_outer^2`
    '''

    if size is None:
        size = (round(2*r_outer+1), round(2*r_outer+1))

    X, Y = divergent(size, centered)

    D = X**2 + Y**2

    return np.logical_and(D <= r_outer**2, D>=r_inner**2).astype(dtype)


def wheel(n_quad, width, size, r=None, start=0, centered=True, dtype=np.bool_):
    '''
    Makes a 2d wheel with specified dtype. If bool or int, can be used as a mask.

    Parameters
    ----------
    n_quad : The number of spokes per quadrant (graph quadrant).
    width : The width of a spoke.
    size : The size of the output array that contains the wheel.
    r : The maximum length of a spoke. Optional.
    start : Offset of the first spoke from 0 in radians.
    centered : If true, the wheel will be centered in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the wheel will be centered
        at the origin pixel (0,0).
    dtype : A valid numpy dtype.

    Returns
    -------
    A wheel that is composed of lines (called spokes) that are evenly rotated
    around the center pixel.
    '''

    wheel = np.zeros(size)
    X, Y = divergent(size, centered)

    for a in np.linspace(start, np.pi-np.pi/(2*n_quad)+start, 2*n_quad):
        mask = np.logical_and(np.sin(a)*X + np.cos(a)*Y < width, np.sin(a)*X + np.cos(a)*Y > -width)
        wheel = np.logical_or(wheel, mask)

    if r:
        wheel = np.logical_and(X**2+Y**2 <= r, wheel)

    return wheel.astype(dtype)
