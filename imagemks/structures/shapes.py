import numpy as np
from math import floor, ceil
from .grids import divergent


def circle(r, size=None, centered=True, dtype=np.bool_):
    '''
    Makes a circle with specified dtype. If bool or int, can be used as a mask.

    Parameters
    ----------
    r : numeric
        The radius of the circle.
    size : tuple, optional
        The size of the output array that contains the circle. Defaults to
        (round(2*r+1), round(2*r+1)).
    centered : boolean, optional
        If true, the circle will be centered in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the circle will be centered
        at the origin pixel (0,0). Defaults to True.
    dtype : object, optional
        A valid numpy dtype. Defaults to boolean.

    Returns
    -------
    circle : ndarray
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
    r_outer : numeric
        The radius of the outer border.
    r_inner : numeric
        The radius of the inner border.
    size : tuple, optional
        The size of the output array that contains the donut. Defaults to
        (round(2*r_outer+1), round(2*r_outer+1)).
    centered : boolean, optional
        If true, the donut will be centered in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the donut will be centered
        at the origin pixel (0,0). Defaults to True.
    dtype : object, optional
        A valid numpy dtype. Defaults to boolean.

    Returns
    -------
    donut : ndarray
        A donut that obeys the equation :math:`r_inner^2 < x^2 + y^2 < r_outer^2`
    '''

    if size is None:
        size = (round(2*r_outer+1), round(2*r_outer+1))

    X, Y = divergent(size, centered)

    D = X**2 + Y**2

    return np.logical_and(D <= r_outer**2, D>=r_inner**2).astype(dtype)


def line(a, size, width=None, r=None, centered=True, dtype=np.bool_):
    X, Y = divergent(size, centered)

    if width:
        line = np.logical_and(np.sin(a)*X + np.cos(a)*Y < width, np.sin(a)*X + np.cos(a)*Y > -width)
    else:
        line = (np.sin(a)*X + np.cos(a)*Y) == 0

    if r:
        line = np.logical_and(X**2+Y**2 <= r, line)

    return line.astype(dtype)


def line_gen(n, size, width=None, r=None, start=0, centered=True, dtype=np.bool_):
    return (line(a, size=size, width=width, r=r) for a in np.linspace(start, np.pi-np.pi/n+start, n))


def wheel(n_quad, size, width=None, r=None, start=0, centered=True, dtype=np.bool_):
    '''
    Makes a 2d wheel with specified dtype. If bool or int, can be used as a mask.

    Parameters
    ----------
    n_quad : int
        The number of spokes per quadrant (graph quadrant).
    size : tuple, optional
        The size of the output array that contains the wheel.
    width : int
        The width of a spoke.
    r : numeric, optional
        The maximum length of a spoke. Optional.
    start : float, optional
        Offset of the first spoke from 0 in radians.
    centered : boolean, optional
        If true, the wheel will be centered in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the wheel will be centered
        at the origin pixel (0,0).
    dtype : object, optional
        A valid numpy dtype.

    Returns
    -------
    wheel : ndarray
        A wheel that is composed of lines (called spokes) that are evenly rotated
        around the center pixel.
    '''

    wheel = np.zeros(size)

    for a in np.linspace(start, np.pi-np.pi/(2*n_quad)+start, 2*n_quad):
        mask = line(a, size=size, width=width, start=start)
        wheel = np.logical_or(wheel, mask)

    if r:
        wheel = np.logical_and(X**2+Y**2 <= r, wheel)

    return wheel.astype(dtype)
