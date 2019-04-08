import numpy as np
import scipy.ndimage as ndi
from ..structures.shapes import circle

def smooth_segmentation(S, r=1, add_cond=0.5, rem_cond=None):
    ''' Smooths a segmented segmentation, where the input is a binary segmentation.
        The add_cond parameter specifies the percentage of neighboring pixels
        that must be in the segmentation for that pixel to be converted to the segmentation.
        Conversely, the rem_parameter specifies the percentage of neighboring pixels outside
        the segmentation that cause a pixel to be removed from the segmentation. r specifies
        the radius of the kernel used to define the neighboring pixels. By default, pixels
        are only added to a segmentation. Alternatively, pixels can be removed and not added.
        Adding and removing does not make sense with this algorithm since a pixel used to
        calculate an add condition could be removed later resulting in a non-smooth segmentation.
        This method can be thought of as a generalization of binary erosion and dilation. '''


    K = circle(1, dtype=np.uint8)
    S = S.astype(np.uint8)

    if add_cond:
        add_cond_met = ndi.convolve(S, K, mode='reflect') > add_cond * np.sum(K)
        to_add = np.logical_and(add_cond_met, np.logical_not(S))

        return np.logical_and(S, to_add)

    elif rem_cond:
        rem_cond_met = ndi.convolve( np.logical_not(S), K, mode='reflect') > rem_cond * np.sum(K)
        to_rem = np.logical_and(rem_cond_met, S)

        return np.logical_and(S, np.logical_not(to_rem))
