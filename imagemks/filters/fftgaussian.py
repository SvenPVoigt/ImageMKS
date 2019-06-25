from ._fftconvolve import fftconvolve2d
from ..structures.weights import gauss

from math import ceil

def fftgauss(img, sigma, theta=0, pad_type=None, **kwargs):
    '''
    Smooths the input image with a gaussian kernel. Uses the fft method and allows
    specifying a custom pad type with **kwargs from numpy.pad documentation. Smoothing
    a color image will smooth each color channel individually.

    Parameters
    ----------
    img : (M,N) or (M,N,3) array
        An image to be smoothed
    sigma : tuple or float
        Tuple defining the standard deviation of the gaussian in x and y directions.
        A single value will assign the same value to x and y st. devs..
    theta : float, optional
        The rotation of the gaussian in radians.
    pad_type : string, optional
        The padding type to be used. For additional information see numpy.pad .
        Defaults to constant.
    kwargs : varies
        See numpy.pad . Defaults to constant_values=0.

    Returns
    -------
    smoothed_image : ndarray
        A smoothed image. Keeps same shape and same number of color channels.

    Notes
    -----
    There are many gaussian smoothing functions. This one is unique because it
    automatically handles color images. It also allows defining very unique
    gaussian kernels with strain and orientation.
    '''
    s = img.shape[:2]
    K = gauss(sigma, theta, size=s)

    return fftconvolve2d(img, K, r=2*ceil(sigma), pad_type=pad_type, centered=True, **kwargs)
