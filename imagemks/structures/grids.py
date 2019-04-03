import numpy as np

def divergent(size, centered=True):
    s0, s1 = size

    X, Y = np.meshgrid(np.arange(-s1//2 + s1%2, s1//2 + s1%2), np.arange(-s0//2 + s0%2, s0//2 + s0%2))

    if centered:
        return (X, Y)

    else:
        return (np.roll(X, -s1//2+s1%2, axis=1), np.roll(Y, -s0//2+s0%2, axis=0))
