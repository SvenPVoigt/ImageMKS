import numpy as np
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk

def make_boundary_image(L, A, thickness=1, color=(255,255,85)):
    ''' L is the labeled image. A is the original image. thickness is the radius of a circle, color is uint8 rgb values.'''

    if A.ndim == 2:
        A = np.stack((A,A,A), axis=2)

    A = np.interp(A, (np.amin(A), np.amax(A)), (0,255)).astype(np.uint8)
    mask = find_boundaries(L)
    mask = binary_dilation(mask, selem=disk(thickness))

    R = A[:,:,0].copy()
    G = A[:,:,1].copy()
    B = A[:,:,2].copy()

    R[mask] = color[0]
    G[mask] = color[1]
    B[mask] = color[2]

    return np.stack((R,G,B), axis=2)
