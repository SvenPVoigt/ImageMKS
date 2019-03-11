import numpy as np
from numpy.fft import fftn, ifftn

from .shapes import make_circle

def local_avg(img, radius, mask=None):
    '''
    This function calculates the local average value using the convolution with a kernel.
    The convolutation is performed in fourier space so it is more efficient for large kernels.
    This function only averages values for the portion of the kernel that overlaps the image.
    This is done by using a mask, which is used to calculate the total number of pixels that the
    kernel covers from the image.
    '''

    shape_in = img.shape
    img = np.pad(img, pad_width=((0,rad+1),(0,rad+1)),
                 mode='constant', constant_values=0)
    mask = np.pad(np.ones(shape_in), pad_width=((0,rad+1),(0,rad+1)),
                 mode='constant', constant_values=0)

    kernel = make_circle(img.shape, radius)


    H1 = fftn(img)
    H2 = fftn(kernel)
    H3 = fftn(mask)

    loc_sum = ifftn(H1*H2.conj()).real
    loc_sum = loc_sum[:shape_in[0],:shape_in[1]]
    loc_norm = ifftn(H3*H2.conj()).real
    loc_norm = loc_norm[:shape_in[0], :shape_in[1]]

    return np.divide(loc_sum, loc_norm)
