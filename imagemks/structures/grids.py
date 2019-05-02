import numpy as np

def divergent(size, centered=True):
    '''
    Creates two grids that specify the x and y values at each point in 2d space.

    Parameters
    ----------
    size : tuple
        The size of the output grids.
    centered : boolean, optional
        If true, the center will be in the middle of the array
        at pixel (size[0]//2, size[1]//2). If false, the center will be
        at the origin pixel (0,0). Defaults to True.

    Returns
    -------
    divergent_grid : tuple of ndarrays
        A grid that defines the x and y values at each point in 2d space. If the
        graph quadrants are recalled, a centered grid will produce an array with
        quadrants ordered as such:

        [ 2 | 1 ]

        [-------]

        [ 3 | 4 ]
    '''
    s0, s1 = size

    X, Y = np.meshgrid(np.arange(-s1//2 + s1%2, s1//2 + s1%2), np.arange(-s0//2 + s0%2, s0//2 + s0%2))

    if centered:
        return (X, Y)

    else:
        return (np.roll(X, -s1//2+s1%2, axis=1), np.roll(Y, -s0//2+s0%2, axis=0))
