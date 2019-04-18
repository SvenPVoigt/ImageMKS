import numpy as np
from math import floor, ceil
from .grids import divergent


def circle(r, size=None, centered=True, dtype=np.float32):
    if size is None:
        size = (round(2*r+1), round(2*r+1))

    X, Y = divergent(size, centered)

    return (X**2 + Y**2 <= r**2).astype(dtype)


def wheel(n_quad, r, size, start=0, centered=True, dtype=np.bool_):
    wheel = np.zeros(size)
    X, Y = divergent(size, centered)

    for a in np.linspace(start, np.pi-np.pi/(2*n_quad)+start, 2*n_quad):
        mask = np.logical_and(np.sin(a)*X + np.cos(a)*Y < r, np.sin(a)*X + np.cos(a)*Y > -r)
        wheel = np.logical_or(wheel, mask)

    return wheel


def oval(a, b, theta, size=None, centered=True, dtype=np.float32):
    pass


def triangle(s, theta, size=None, centered=True, dtype=np.float32):
    ''' A regular triangle with all sides equal '''
    pass


def rectangle(shape, theta, size=None, centered=True, dtype=np.float32):
    pass
