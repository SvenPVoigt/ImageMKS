import numpy as np
import scipy.ndimage as ndi
from ..structures.shapes import circle

def smooth_binary(S, r=1, add_cond=0.5, rem_cond=None):
    '''
    Smooths a segmented image, where the input is a binary segmentation.

    Parameters
    ----------
    S : ndarray of booleans
        A binary image.
    add_cond : float, optional
        Parameter that specifies the percentage of neighboring pixels
        that must be in the segmentation for that pixel to be converted to
        the segmentation.
    rem_cond : float, optional
        Parameter that specifies the percentage of neighboring pixels
        outside the segmentation that cause a pixel to be removed from
        the segmentation.
    r : numeric, optional
        The radius of the kernel used to define the neighboring pixels.

    Returns
    -------
    smooth_segmentation : ndarray of booleans
        A smoothed version of the segmented image.

    Notes
    -----
    By default, pixels are only added to a segmentation. Alternatively, pixels
    can be removed and not added. Adding and removing does not make sense with
    this algorithm since a pixel used to calculate an add condition could be
    removed later resulting in a non-smooth segmentation. This method can be
    thought of as a generalization of binary erosion and dilation.
    '''

    K = circle(r, dtype=np.uint8)
    S = S.astype(np.uint8)

    if add_cond:
        add_cond_met = ndi.convolve(S, K, mode='reflect') > \
                       round( add_cond * np.sum(K) )

        return add_cond_met

    elif rem_cond:
        rem_cond_met = ndi.convolve( np.logical_not(S), K, mode='reflect') > \
                       round( rem_cond * np.sum(K) )

        return np.logical_not(rem_cond_met)
