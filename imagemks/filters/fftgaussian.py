from ._fftconvolve import fftconvolve2d
from ..structures.weights import gauss

def fftgauss(img, sigma):
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

    # Performing the convolution of image with gaussian kernel
    H1 = fftn(img)
    H2 = fftn(kernel).conj()
    conv = np.fft.fftshift(ifftn(H1*H2)).real

    # Returning the convolution as the original size of the image
    return conv[2*sigma:ogS[0]+2*sigma,2*sigma:ogS[1]+2*sigma]


def fftder1gauss(img, sigma):
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
