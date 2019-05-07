import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift

def maskfft(image, mask, centered=True):
    if image.ndim == 3:
        mask = np.stack((mask,mask,mask), axis=2)

    H_low = fftn(image, axes=(0,1))

    if centered:
        H_low = fftshift(H_low, axes=(0,1))

    H_high = H_low.copy()

    H_low[np.logical_not(mask)] = 0
    H_high[mask] = 0

    if centered:
        H_low = ifftshift(H_low)
        H_high = ifftshift(H_high)

    F_low = ifftn(H_low, axes=(0,1)).real
    F_high = ifftn(H_high, axes=(0,1)).real

    return (F_low, F_high)
