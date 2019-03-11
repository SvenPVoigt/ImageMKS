import numpy as np
import scipy.ndimage as ndi
from skimage.measure import regionprops

def rem_obj(img, size, gre_or_less='less'):
    labels, n = ndi.label(img)
    sizes = ndi.sum(img, labels, range(n + 1))

    if gre_or_less == 'less':
        rem_region = sizes < size
    elif gre_or_less == 'gre':
        rem_region = sizes > size
    else:
        raise ValueError('gre_or_less must be gre or less')
        return None

    remove_pixel = rem_region[label_im]
    img[remove_pixel] = 0

    return img


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
