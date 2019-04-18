from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


def make_label_colors(labels, cmap='tab20'):
    ''' For some reason matplotlib on PACE does not have tab20. Reason could be that it is verson 2.0,
        and current version is 2.2. Usual code: cols = [plt.cm.tab20(i) for i in range(20)] '''

    if cmap is 'tab20':
        color_list =   [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
                        (0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0),
                        (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
                        (1.0, 0.7333333333333333, 0.47058823529411764, 1.0),
                        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
                        (0.596078431372549, 0.8745098039215686, 0.5411764705882353, 1.0),
                        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
                        (1.0, 0.596078431372549, 0.5882352941176471, 1.0),
                        (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
                        (0.7725490196078432, 0.6901960784313725, 0.8352941176470589, 1.0),
                        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
                        (0.7686274509803922, 0.611764705882353, 0.5803921568627451, 1.0),
                        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
                        (0.9686274509803922, 0.7137254901960784, 0.8235294117647058, 1.0),
                        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
                        (0.7803921568627451, 0.7803921568627451, 0.7803921568627451, 1.0),
                        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
                        (0.8588235294117647, 0.8588235294117647, 0.5529411764705883, 1.0),
                        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
                        (0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0)]

    else:
        raise ValueError('Specified cmap not supported')

    l = len(labels)
    cols = [(0.07,0.07,0.07,0.07),] + color_list * (l // 20)
    cols = cols + cols[:((l % 20) - 1)]

    return LinearSegmentedColormap.from_list('cols', cols)


def plot_cells(img, title=None, cmap='Greys', peaks=None, axes_off=True, figsize=(9,6)):
    fig, axs = plt.subplots(1,1, figsize=figsize)
    im = axs.imshow(img, cmap=cmap)
    if axes_off:
        axs.axis('off')
    if title is not None:
        axs.set_title(title)
    if peaks is not None:
        axs.plot(peaks[:,1], peaks[:,0], 'r.')
    plt.show()


def draw_hist_Mean_SD(img, nbins, range_hist=None, std_weight=2, figsize=(9,6)):
    hist, edges = np.histogram(img.ravel(), bins=nbins, range=range_hist)
    x = 0.5 * (edges[:-1] + edges[1:])
    fig, axs = plt.subplots(1,1, figsize=figsize)
    axs.plot(x, hist)
    hist_avg = np.average(img)
    hist_std = np.std(img)
    hist_min = hist_avg - std_weight*hist_std
    hist_max = hist_avg + std_weight*hist_std
    axs.axvline(x=hist_avg, color='red')
    axs.axvline(x=hist_min, color='blue')
    axs.axvline(x=hist_max, color='blue')
    plt.show()


def plot_2d_stats(A, title=None, figsize=(9,6)):
    fig, axs = plt.subplots(1,1, figsize=figsize)

    im = axs.imshow(A, extent=[-A.shape[1]//2, A.shape[1]//2 + A.shape[1]%2,
                               -A.shape[0]//2, A.shape[0]//2 + A.shape[0]%2])

    if title is None:
        title = '2-point Statistics'

    axs.set_title(title)

    plt.show()


def overlay_binary(img, img_binary, cmap_img=None, col_binary=None, figsize=(9,6), axes_off=True, title=None):
    if col_binary is None:
        col_binary = 'black'

    col1 = colorConverter.to_rgba('white')
    col2 = colorConverter.to_rgba(col_binary)

    binary_cmap = mpl.colors.LinearSegmentedColormap.from_list('binary_cmap',[col1,col2],2)

    binary_cmap._init()
    alphas = (0,1,1,1,1)
    binary_cmap._lut[:,-1] = alphas

    fig, axs = plt.subplots(1,1, figsize=figsize)
    axs.imshow(img, cmap=cmap_img)
    axs.imshow(img_binary, cmap=binary_cmap)
    if axes_off:
        axs.axis('off')
    if title is not None:
        axs.set_title(title)
    plt.show()


def boxed_circle(rad, shape):
    x, y = np.ogrid[ -shape[0]//2:shape[0]//2 + shape[0]%2,
                     -shape[1]//2:shape[1]//2 + shape[1]%2 ]

    return ( x**2 + y**2 <= rad**2 ).astype(np.float64)
