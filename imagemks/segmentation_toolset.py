import numpy as np
from numpy.fft import fftshift
from mkl_fft import fftn, ifftn
import rdf_cells as rdf
import kernels_cells as kernels

from scipy import ndimage
from skimage.measure import regionprops

def gBlur(img, sigma):
    ogS = img.shape

    # Padding Image
    img = np.pad(img, pad_width = ((2*sigma,2*sigma),(2*sigma,2*sigma)),mode = 'edge')

    # Defining the space for a gaussian kernel
    y = img.shape[0]
    ys = np.linspace(-y/2, y/2, y)
    x = img.shape[1]
    xs = np.linspace(-x/2, x/2, x)

    # Making the 2D gaussian kernel by taking two normDF in x and y and taking the outer product
    yPDF = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-ys**(2)/(sigma**2))
    xPDF = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-xs**(2)/(sigma**2))
    kernel = np.outer(yPDF,xPDF)

    # y, x = np.ogrid[0:ySize, 0:xSize]
    # y = y - ySize//2
    # y = np.roll(y, ySize//2, axis = 0)
    # ys = y.ravel()
    # x = x - xSize//2
    # x = np.roll(x, xSize//2, axis = 1)
    # xs = x
    #
    # # Making the 2D gaussian kernel by taking two normDF in x and y and taking the outer product
    # yPDF = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-ys**(2)/(sigma**2))
    # xPDF = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-xs**(2)/(sigma**2))
    # kernel = np.outer(yPDF,xPDF)

    # Performing the convolution of image with gaussian kernel
    H1 = fftn(img)
    H2 = fftn(kernel).conj()
    conv = np.fft.fftshift(ifftn(H1*H2)).real

    # Returning the convolution as the original size of the image
    return conv[2*sigma:ogS[0]+2*sigma,2*sigma:ogS[1]+2*sigma]


def gGrad(img, sigma):
    ogS = img.shape
    img = np.pad(img, pad_width = ((2*sigma,2*sigma),(2*sigma,2*sigma)),mode = 'edge')

    # Defining the space for a gaussian kernel
    y = img.shape[0]
    ys = np.linspace(-y/2, y/2, y)
    x = img.shape[1]
    xs = np.linspace(-x/2, x/2, x)

    # Making the 2D partial derivative gaussian kernels
    yPDF = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-ys**(2)/(sigma**2))
    xPDF = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-xs**(2)/(sigma**2))
    dyPDF = (-ys / (np.sqrt(2*np.pi)*sigma**3)) * np.exp(-ys**(2)/(sigma**2))
    dxPDF = (-xs / (np.sqrt(2*np.pi)*sigma**3)) * np.exp(-xs**(2)/(sigma**2))
    dykernel = np.outer(dyPDF,xPDF)
    dxkernel = np.outer(yPDF,dxPDF)

    # FFT of img
    H1 = fftn(img)

    # Gradient in y direction
    H2 = fftn(dykernel).conj()
    yconv = np.fft.fftshift(ifftn(H1*H2)).real
    yconv = yconv[2*sigma:ogS[0]+2*sigma,2*sigma:ogS[1]+2*sigma]

    # Gradient in x direction
    H2 = fftn(dxkernel).conj()
    xconv = np.fft.fftshift(ifftn(H1*H2)).real
    xconv = xconv[2*sigma:ogS[0]+2*sigma,2*sigma:ogS[1]+2*sigma]

    # Returning the magnitude of the gradient
    return np.sqrt(yconv**2 + xconv**2)


def local_avg(img, rad, mask=None):
    ogS = img.shape
    ex_pad = 1
    img = np.pad(img, pad_width=((0,rad+ex_pad),(0,rad+ex_pad)),
                 mode='constant', constant_values=0)

    ySize = img.shape[0]
    xSize = img.shape[1]

    y, x = np.ogrid[0:ySize, 0:xSize]
    y = y - ySize//2
    y = np.roll(y, ySize//2, axis = 0)
    x = x - xSize//2
    x = np.roll(x, xSize//2, axis = 1)

    c_kernel = (x**2 + y**2 <= rad**2).astype(np.float64)

    if mask is None:
        norm = np.ones(ogS)
    else:
        norm = mask.copy()

    norm = np.pad(norm, pad_width=((0,rad+ex_pad),(0,rad+ex_pad)),
                  mode='constant', constant_values=0)

    H1 = fftn(img)
    H2 = fftn(c_kernel)
    H3 = fftn(norm)

    loc_sum = ifftn(H1*H2.conj()).real
    # loc_sum = loc_sum[rad+ex_pad:ogS[0]+rad+ex_pad, rad+ex_pad:ogS[1]+rad+ex_pad]
    loc_sum = loc_sum[:ogS[0],:ogS[1]]
    loc_norm = ifftn(H3*H2.conj()).real
    # loc_norm = loc_norm[rad+ex_pad:ogS[0]+rad+ex_pad, rad+ex_pad:ogS[1]+rad+ex_pad]
    loc_norm = loc_norm[:ogS[0], :ogS[1]]

    if mask is None:
        loc_avg = np.divide(loc_sum, loc_norm)
    else:
        loc_avg = np.divide(loc_sum, loc_norm, where=loc_norm!=0) * (loc_norm>0)

    return (loc_avg, loc_sum, loc_norm)


def conv_dilate(img, rad):
    rad_intern = int( round( rad ) )
    rad_intern += 1
    ogS = img.shape
    img = np.pad(img, pad_width=( (rad_intern,rad_intern),(rad_intern,rad_intern) ), mode='constant', constant_values=0)

    y, x = img.shape
    ys, xs = np.ogrid[-y/2:y/2:1,-x/2:x/2:1]

    c_kernel = (xs**2 + ys**2 <= (rad)**2).astype(np.float64)

    shift_to_mid_y = c_kernel.shape[0]//2 + 1
    shift_to_mid_x = c_kernel.shape[1]//2 + 1
    c_kernel = np.roll(c_kernel, shift_to_mid_y, axis = 0)
    c_kernel = np.roll(c_kernel, shift_to_mid_x, axis = 1)

    # FFT of img
    H1 = fftn(img)

    H2 = fftn(c_kernel).conj()
    bin_dil = ifftn(H1*H2).real
    bin_dil = bin_dil[rad_intern:ogS[0]+rad_intern,rad_intern:ogS[1]+rad_intern]

    return bin_dil


def mk_circle(radius_px):
    ys, xs = np.ogrid[-radius_px:radius_px+1:1,-radius_px:radius_px+1:1]
    c_kernel = (xs**2 + ys**2 <= radius_px**2).astype(np.float64)
    return c_kernel


def do_rem_obj(img, size, gre_or_less='less', rem_add='rem'):
    img_rem = img.copy()
    label_im, nb_labels = ndimage.label(img_rem)
    sizes_matrix = ndimage.sum(img_rem, label_im, range(nb_labels + 1))
    if gre_or_less == 'less':
        rem_region = sizes_matrix < size
    elif gre_or_less == 'gre':
        rem_region = sizes_matrix > size
    else:
        raise ValueError('gre_or_less must be gre or less')
        return None
    remove_pixel = rem_region[label_im]
    if rem_add == 'rem':
        img_rem[remove_pixel] = 0
    elif rem_add == 'add':
        img_rem[remove_pixel] = 1
    else:
        raise ValueError('rem_add must be rem or add')
        return None
    return (img_rem, label_im)


def remove_freq(img, cutmax=90, cutmin=None):
    H1 = img.copy()
    H1 = fftn(H1)
    ySize = H1.shape[0]
    xSize = H1.shape[1]

    y, x = np.ogrid[0:ySize, 0:xSize]
    y = y - ySize//2
    y = np.roll(y, ySize//2, axis = 0)
    x = x - xSize//2
    x = np.roll(x, xSize//2, axis = 1)

    H1[x**2 + y**2 > cutmax**2] = 0

    if cutmin is not None:
        H1[x**2 + y**2 < cutmin**2] = 0

    return np.real(ifftn(H1))


def gen_marks(label_img):
    props = regionprops(label_img)
    markers = np.zeros(label_img.shape)
    label_val = 1

    for i in props:
        x, y = i.centroid
        x = int(round(x))
        y = int(round(y))
        markers[x,y] = label_val
        label_val += 1

    return markers


def th_freq_1(img, num_bins=100):
    imgTh = img.copy()
    minbin = np.amin(imgTh)
    maxbin = np.amax(imgTh)
    bins = np.linspace(minbin, maxbin, num_bins)
    hist, bins = np.histogram(imgTh.ravel(), bins = bins)

    bins_mid = (bins[1:] + bins[:-1])/2
    big_peak = bins_mid[np.argmax(hist)]
    th = 0.5 + big_peak

    return th


def adjust_hist(img, std_weight=2):
    img_out = img.copy()
    hist_avg = np.average(img_out)
    hist_std = np.std(img_out)
    hist_min = hist_avg - std_weight*hist_std
    hist_max = hist_avg + std_weight*hist_std
    img_out[img_out > hist_max] = hist_max
    img_out[img_out < hist_min] = hist_min
    return img_out


def find_change(y, smooth_size_pix=None):
    der1 = np.gradient(y)
    der2 = np.gradient(der1)

    x = np.arange(len(der2))

    if smooth_size_pix is not None:
        der2 = kernels.kernel_conv_1d(smooth_size_pix, x, der2, kernel_type='gaussian', periodic=False)

    return np.argmax(der2)


def get_size(img, blur_radius=3, std_weight=3, search_cutoff=0.25, pix_res=5, der_smooth_pix=5):
    img = gBlur(img, blur_radius)
    img = adjust_hist(img, std_weight)

    A = fftn(img)
    A = np.conj(A) * A
    A = ifftn(A).real

    x, y = rdf.stats_to_rdf(stats_norm=fftshift(A))

    largest_considered_cell_radius = search_cutoff * max(img.shape)
    x = x[x < largest_considered_cell_radius]
    y = y[:len(x)]

    x_n = kernels.linspace_steps(x[0], x[-1], pix_res)
    y_n = np.interp(x_n, x, y)

    ind = find_change(y_n, der_smooth_pix)

    return x_n[ind]
