import numpy as np
from math import floor, ceil
from .grids import divergent


def circle(r, size=None, centered=True, dtype=np.float32):
    if size is None:
        size = (4*floor(r)+1, 4*floor(r)+1)
        # size = (round(2*r+1), round(2*r+1))

    X, Y = divergent(size, centered)

    return (X**2 + Y**2 <= r**2).astype(dtype)


def oval(a, b, theta, size=None, centered=True, dtype=np.float32):
    pass


def triangle(s, theta, size=None, centered=True, dtype=np.float32):
    ''' A regular triangle with all sides equal '''
    pass


def rectangle(shape, theta, size=None, centered=True, dtype=np.float32):
    pass
