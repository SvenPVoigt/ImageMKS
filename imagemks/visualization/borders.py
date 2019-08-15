import numpy as np
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk

def make_boundary_image(L, A, thickness=1, color=(255,255,85), rescale_hist=True):
    '''
    Marks borders of segmentation on the original image so that borders can
    be evaluated. Similar to skimage.segmentation.mark_boundaries. This function
    allows specifying a border thickness. This makes borders visible in high
    resolution images.

    Parameters
    ----------
    L : (M,N) array of dtype long
        The labeled image that is a segmentation of A.
    A : (M,N) or (M,N,3) array of any numerical dtype
        The original image. Grayscale and color are supported.
        thickness : Thickness of the borders in pixels. Default is 1.
        color : Tuple of 3 uint8 RGB values.

    Returns
    -------
    marked_image : ndarray
        The original image with marked borders, 3 color channels.
    '''

    if A.ndim == 2:
        A = np.stack((A,A,A), axis=2)

    if rescale_hist:
        A = np.interp(A, (np.amin(A), np.amax(A)), (0,255)).astype(np.uint8)
    else:
        A = A.astype(np.uint8)

    mask = find_boundaries(L)
    mask = binary_dilation(mask, selem=disk(thickness))

    R = A[:,:,0].copy()
    G = A[:,:,1].copy()
    B = A[:,:,2].copy()

    R[mask] = color[0]
    G[mask] = color[1]
    B[mask] = color[2]

    return np.stack((R,G,B), axis=2)
