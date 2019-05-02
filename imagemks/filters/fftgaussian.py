from ._fftconvolve import fftconvolve2d
from ..structures.weights import gauss

def fftgauss(img, sigma, theta=0, pad_type=None, **kwargs):
    '''
    Smooths the input image with a gaussian kernel. Uses the fft method and allows
    specifying a custom pad type with **kwargs from numpy.pad documentation. Smoothing
    a color image will smooth each color channel individually.

    Parameters
    ----------
    img : An image to be smoothed of size (M,N) or (M,N,3).
    sigma : Tuple defining the standard deviation of the gaussian in x and y directions.
        A single value will assign the same value to x and y st. devs..
    theta : The rotation of the gaussian in radians.
    pad_type : The padding type to be used. For additional information see numpy.pad .
        Defaults to constant.
    kwargs : See numpy.pad . Defaults to constant_values=0.

    Returns
    -------
    A smoothed image.

    Notes
    -----
    There are many gaussian smoothing functions. This one is unique because it
    automatically handles color images. It also allows defining very unique
    gaussian kernels with strain and orientation.
    '''
    s = img.shape
    K = gauss(sigma, theta, size=s)

    return fftconvolve2d(img, K, r=2*sigma, pad_type=pad_type, centered=True, **kwargs)
