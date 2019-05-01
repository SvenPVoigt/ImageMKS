import numpy as np

def fftconvolve2d(h1, h2, r, pad_type, centered=True, **kwargs):
    '''
    This method adds to the functionality of scipy.signal.fftconvolve
    by providing padding options for 2D arrays.

    Parameters
    ----------
    h1 : The signal to be convolved, should be (M,N) and is usually the image.
    h2 : The signal to convolve with, should be (M,N) and is usually the filter.
        Will always be padded with zeros.
    r : The padding width. Termed r since an int value will introduce a radius
        around the input. Different padding widths not supported.
    pad_type : The padding type to be used. For additional information see numpy.pad .
    centered : Needed since the filters in this package can be defined as centered
        or origin type.
    kwargs : see numpy.pad

    Results
    -------
    a = A convolved signal in real space. Imaginary outputs not supported.
    '''

    assert(h1.ndim==2 and h2.ndim==2)
    assert(h1.shape==h2.shape)
    # Check that r is an int. Other pad widths not supported bc of return slicing.
    assert(isinstance(r,int))

    s0, s1 = h1.shape

    h1 = np.pad(h1, pad_width=r, mode=pad_type, **kwargs)

    if not centered:
        h2 = np.fft.fftshift(h2)

    h2 = np.pad(h2, pad_width=r, mode='constant', constant_values=0)

    # Performing the convolution of image with the kernel
    H1 = fftn(h1)
    H2 = fftn(h2).conj()
    conv = np.fft.fftshift(ifftn(H1*H2)).real

    return conv[r:s0+r,r:s1+r]
