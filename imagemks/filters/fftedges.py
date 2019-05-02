import numpy as np
from ._fftconvolve import fftconvolve2d
from ..structures import circle

def local_avg(img, rad, mask=None):
    '''
    Returns the local average of a neighborhood taking into account that the
    edges need to be treated differently. Averages are only calculated at the
    edges using values inside the image and the neighborhood.

    Parameters
    ----------
    img : An image.
    rad : The radius of the neighborhood.
    mask : A binary array that can be used to define what is outside the
        image. Optional.

    Returns
    -------
    The local averages at all pixel locations.
    '''
    s = img.shape

    if mask is None:
        counts = np.ones(s)
    else:
        counts = mask.copy()

    pad_r = rad+1

    K = circle(rad, size=s)

    sums = fftconvolve2d(img, K, r=pad_r)
    norms = fftconvolve2d(counts, K, r=pad_r)

    return np.divide(sums, norms)
