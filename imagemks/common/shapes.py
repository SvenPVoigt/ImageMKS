import numpy as np

def make_circle(shape, radius):
    y, x = np.ogrid[0:shape[0], 0:shape[1]]
    y = y - shape[0]//2
    y = np.roll(y, shape[0]//2, axis = 0)
    x = x - shape[1]//2
    x = np.roll(x, shape[1]//2, axis = 1)

    return (x**2 + y**2 <= radius**2).astype(np.float64)
