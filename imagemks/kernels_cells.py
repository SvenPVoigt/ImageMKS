import numpy as np

def epanechnikov_kernel(bin_space, width):
    #epan = 0.75*(5**(-0.5))*(1 - (bin_space**2)/(5*width**2))/width

    epan = (1 - (bin_space**2)/(width**2))

    # This approach might not work if the negative values become too large
    # Will require only defining kernel within width
    epan[epan<0] = 0
    return epan


def area_kernel(bin_space, width):
    ''' This kernel converts a single point in space to the area of intersection
        with a plane at r.'''
    area = np.zeros(len(bin_space))
    indices = np.abs(bin_space) <= width
    area[indices] = np.pi * (width**2 - bin_space[indices]**2)

    return area


def gaussian_kernel(bin_space, sigma):
    gauss = np.exp(-bin_space**(2)/(sigma**2))
    return gauss


def check_bin_spacing(bins):
    ''' The bin spacing gets checked '''
    steps = bins[1:] - bins[:-1]
    if not np.all(np.isclose(steps[0], steps)):
        raise ValueError('Bins need to be same width!')
        return None
    return steps[0]


def linspace_steps(start, end, step_size):
    if (end-start)%step_size != 0:
        end = round(end + step_size - (end-start)%step_size)
    n = int((end-start)//(step_size) + 1)
    return np.linspace(start, end, n)


def make_kernel(width_real, bins, kernel_type='epanechnikov'):
    ''' Makes a kernel for a regular binned space. Width of kernel
        should be defined in real space not number of bins.'''

    step = check_bin_spacing(bins)
    del step

    bins = np.roll(bins - bins[len(bins)//2], -(len(bins)//2))

    if kernel_type == 'epanechnikov':
        kernel = epanechnikov_kernel(bin_space=bins, width=width_real)
    elif kernel_type == 'gaussian':
        kernel = gaussian_kernel(bin_space=bins, sigma=width_real)
    elif kernel_type == 'area':
        kernel = area_kernel(bin_space=bins, width=width_real)
    else:
        raise ValueError('Entered kernel type not supported!')

    return kernel


def kernel_conv_1d(width_real, r, counts, kernel_type='epanechnikov', periodic=True):
    ''' r is the center of the bins '''
    step = check_bin_spacing(r)

    if periodic:
        kernel = make_kernel(width_real, r, kernel_type)
        h1 = np.fft.fftn(kernel)
        h2 = np.fft.fftn(counts)
        density = np.fft.ifftn(h1.conj() * h2).real
        return density / width_real

    elif not periodic:
        ###### The bins and counts are overwritten by the corresponding padded bins and counts for
        ###### the non-periodic case. The original size is stored to return the same size array.
        og_size = len(r)
        counts = np.pad(counts, (0,og_size), 'constant', constant_values=0)
        r = np.linspace(0, 2*r[-1]+step, num=2*og_size)

        kernel = make_kernel(width_real, r, kernel_type)

        h1 = np.fft.fftn(kernel)
        h2 = np.fft.fftn(counts)
        density = np.fft.ifftn(h1.conj() * h2).real

        ###### Not sure it is necessary to normalize.....
        norm = np.concatenate((np.ones(og_size),np.zeros(og_size)))
        h3 = np.fft.fftn(norm)
        norm = np.fft.ifftn(h1.conj() * h3).real
        norm = norm

        return density[:og_size] / norm[:og_size]

    else:
        raise ValueError('Input variable \'periodic\' needs to be boolean!')
