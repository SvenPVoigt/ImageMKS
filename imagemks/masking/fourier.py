import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift

def maskfourier(image, mask, centered=True):
    '''
    Masks the fourier transform of an image using the given mask.

    Parameters
    ----------
    image : An (M,N) array or (M,N,3) array.
    mask : An (M,N) array. Will be expanded to 3 channels if image is color.
    centered : Is the mask for a centered fourier transform? Optional.

    Returns
    -------
    (A, B) : Two images generated from the masked fourier transforms. A contains
        an image reconstructed from values in the mask. B contains an image
        reconstructed from values outside the mask.

    Examples
    --------
    >>> import numpy as np
    >>> from imagemks.masking import maskfourier
    >>> image = np.ones((3,3))
    >>> mask_centered = np.array([[0,1,0],
                                  [1,1,1],
                                  [0,1,0]])

    >>> mask_origin = np.fft.ifftshift(mask_centered)
    >>> mask_origin
    array([[1, 1, 1],
           [1, 0, 0],
           [1, 0, 0]])

    The difference between centered masks and uncentered can be resolved by
    using np.fft.fftshift and np.ifftshift. By default all masks in structures
    are centered.

    >>> maskfourier(image, mask_centered, centered=True)
    (array([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]),
     array([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]]))
    '''

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

    F_low = ifftn(H_low).real
    F_high = ifftn(H_high).real

    return (F_low, F_high)
