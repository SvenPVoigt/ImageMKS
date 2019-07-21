# Importing general libraries
import numpy as np
import scipy.ndimage as ndi
import pandas as pd

# Importing specific functions
from skimage.morphology import remove_small_objects, watershed
from skimage.feature import corner_peaks
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops
from skimage.color import label2rgb

# Importing ImageMKS functions
from ..filters import fftgauss, local_avg
from ..structures import donut
from ..masking import maskfourier
from ..visualization import make_boundary_image



def _gen_marks(label_img):
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


def default_parameters(cell_type):
    '''
    Generates a dictionary of default paramaters.

    Parameters
    ----------
    cell_type : string
        Either muscle or stem. More support coming soon.

    Returns
    -------
    params : dictionary
        Params defines smooth_size, intensity_curve, short_th_radius, long_th_radius,
        min_frequency_to_remove, max_frequency_to_remove,
        max_size_of_small_objects_to_remove, power_adjust, peak_min_distance,
        size_after_watershed_to_remove, and cyto_local_avg_size.
    '''
    if cell_type is 'muscle':
        params = {
            'smooth_size': 3,
            'intensity_curve': 2,
            'short_th_radius': 50,
            'long_th_radius': 600,
            'min_frequency_to_remove': 1,
            'max_frequency_to_remove': 500,
            'max_size_of_small_objects_to_remove': 300,
            'power_adjust': 1,
            'peak_min_distance': 10,
            'size_after_watershed_to_remove': 300,
            'cyto_local_avg_size': 200
            }

        return params

    elif cell_type is 'stem':

        params = {
            'smooth_size': 3,
            'intensity_curve': 3,
            'short_th_radius': 100,
            'long_th_radius': 800,
            'min_frequency_to_remove': 1,
            'max_frequency_to_remove': 500,
            'max_size_of_small_objects_to_remove': 1100,
            'power_adjust': 1,
            'peak_min_distance': 10,
            'size_after_watershed_to_remove': 1100,
            'cyto_local_avg_size': 200
            }

        return params

    else:
        print('Sorry this cell type is not yet supported.')


def segment_fluor_cells(N, C, smooth_size, intensity_curve, short_th_radius,
                long_th_radius, min_frequency_to_remove, max_frequency_to_remove,
                max_size_of_small_objects_to_remove, peak_min_distance,
                size_after_watershed_to_remove, cyto_local_avg_size, zoomLev):
    '''
    Segments fluorescent cells.

    Parameters
    ----------
    N : (M,N) numpy array
        An image of nuclei with size (M,N)
    C : (M,N) numpy array
        An image of the cytoskeleton with same size as the nucleus image (M,N).
    smooth_size : int, pixels
        The sigma of the gaussian.
    intensity_curve : int
        Exponent of the curve used to fit intensities on range [0,1]
    short_th_radius : int, pixels
        Radius of neighborhood used to calculate a local
        average threshold.
    long_th_radius : int, pixels
        Radius of neighborhood used to calculate a local
        average threshold
    min_frequency_to_remove : int, pixels
        Frequency in pixels used to define donut mask.
    max_frequency_to_remove : int, pixels
        Frequency in pixels used to define donut mask.
    max_size_of_small_objects_to_remove : float, micrometers^2
        Size beneath which no cells can exist.
    peak_min_distance : int, pixels
        Min distance between nuclei.
    size_after_watershed_to_remove : float, micrometers^2
        Size beneath which no cells can exist.
        Calculated after watershed.
    cyto_local_avg_size : int, pixels
        Radius of neighborhood used to calculate a local
        average threshold
    zoomLev : int
        Real magnification of the image.

    Returns
    -------
    (N, C) : list of (M,N) numpy arrays. Long dtype
        N is a labeled nucleus image. Where each label corresponds to an individual
        cell. 0 corresponds to the background. C is a labeled cytockeleton image. The
        labels correspond to the closest nucleus in N.
    '''

    # Processing color images and converting the images to the range [0,1]
    if N.ndim == 3:
        N = np.sum(np.array(N), axis=2)

    N = (( N-np.amin(N)) / np.ptp(N))

    if C.ndim == 3:
        C = np.sum(np.array(C), axis=2)

    C = (( C-np.amin(C)) / np.ptp(C))

    # Preprocessing by smoothing and scaling nucleus image
    N = fftgauss(N, smooth_size, pad_type='edge')
    N = np.power(N, intensity_curve)

    # Local avg threshold
    th_short = N > local_avg(N, short_th_radius)
    th_long = N > local_avg(N, long_th_radius)
    del th_short, th_long
    th_N = (th_short*th_long)

    # Remove small objects
    th_N = remove_small_objects(th_N, max_size_of_small_objects_to_remove * (zoomLev))

    # Distance transform
    distance = ndi.distance_transform_edt(th_N)

    # Mark the maxima in the distance transform and assign labels
    peak_markers = corner_peaks(distance, min_distance=peak_min_distance, indices=False)
    peak_markers = ndi.label(peak_markers)[0]

    # Separate touching nuclei using the watershed markers, removing small objects, and relabeling
    label_N = watershed(th_N, peak_markers, mask=th_N)
    label_N = remove_small_objects(label_N, size_after_watershed_to_remove * (zoomLev))

    # Assigning sequential labels
    old_labels = np.unique(label_Ni)
    for i in range(len(old_labels)):
        label_Ni[label_Ni == old_labels[i]] = i

    # Step 14: local threshold of the cytoskeleton
    th_C = C > local_avg(C, cyto_local_avg_size)[0]

    # Step 15: generate relabeled markers from the nuclei centroids
    new_markers = _gen_marks(label_N)

    # Step 16: watershed of cytoskeleton using new_markers
    label_C = watershed(th_C, new_markers, mask=th_C.astype(np.bool_))

    return [label_N, label_C]


def measure_fluor_cells(label_Nuc, label_Cyto, pix_size):
    '''
    Generates measurements for labeled Nucleus images and labeled Cytoskeleton images.

    Parameters
    ----------
    label_Nuc : (M,N) long dtype
        A labeled nucleus image. Where each label corresponds to an individual
        cell. 0 corresponds to the background.
    label_Cyto : (M,N) long dtype
        A labeled cytockeleton image. The labels correspond to the closest
        nucleus in N. 0 corresponds to the background.

    Returns
    -------
    Measurements : dataframe of measurements for each cell
        Cell_Number, Nuc_Area_um2, Nuc_Perimeter_um, Nuc_Area_Factor,
        Nuc_Major_L_um, Nuc_Minor_L_um, Nuc_eccentricity, Nuc_orientation,
        Nucleus_eq_diameter_um, Cyto_Area_um2, Cyto_um, Cyto_Area_Factor,
        Cyto_orientation, Cyto_Major_L_um, Cyto_Minor_L_um
    '''

    nuc_props = regionprops(label_Nuc)
    cyto_props = regionprops(label_Cyto)

    cell_index = 1

    col_names = ['Cell_Number', 'Nuc_Area_um2', 'Nuc_Perimeter_um', 'Nuc_Area_Factor',
                 'Nuc_Major_L_um', 'Nuc_Minor_L_um', 'Nuc_eccentricity', 'Nuc_orientation',
                 'Nucleus_eq_diameter_um', 'Cyto_Area_um2', 'Cyto_um', 'Cyto_Area_Factor',
                 'Cyto_orientation', 'Cyto_Major_L_um', 'Cyto_Minor_L_um']

    prop_df = pd.DataFrame(columns = col_names)

    for i in range(len(nuc_props)):
        nucleus_A = nuc_props[i].area / (pix_size**2)
        nucleus_P = nuc_props[i].perimeter / pix_size
        nucleus_SF = nucleus_A / (nucleus_P**2)

        cyto_A = cyto_props[i].area  / (pix_size**2)
        cyto_P = cyto_props[i].perimeter  / pix_size
        cyto_SF = cyto_A / (cyto_P**2)

        prop_df = prop_df.append(pd.DataFrame(
                      {'Cell_Index': (cell_index,),
                       'Nuc_Area_um2': (nucleus_A,),
                       'Nuc_Perimeter_um': (nucleus_P,) ,
                       'Nuc_Area_Factor': (nucleus_SF,),
                       'Nuc_Major_L_um': (nuc_props[i].major_axis_length,),
                       'Nuc_Minor_L_um': (nuc_props[i].minor_axis_length,),
                       'Nuc_eccentricity': (nuc_props[i].eccentricity,),
                       'Nuc_orientation': (nuc_props[i].orientation,),
                       'Nucleus_eq_diameter_um': (nuc_props[i].equivalent_diameter,),
                       'Cyto_Area_um2': (cyto_A,),
                       'Cyto_Perimeter_um': (cyto_P,),
                       'Cyto_Area_Factor': (cyto_SF,),
                       'Cyto_orientation': (cyto_props[i].orientation,),
                       'Cyto_Major_L_um': (cyto_props[i].major_axis_length,),
                       'Cyto_Minor_L_um': (cyto_props[i].minor_axis_length,)}))
        cell_index += 1

    # Reordering the columns
    prop_df = prop_df[col_names]


    return prop_df


def visualize_fluor_cells(L, A, thickness=1):
    '''
    Colors the original image with the segmented image. Also marks borders of
    segmentation on the original image so that borders can be evaluated.

    Parameters
    ----------
    L : (M,N) long dtype
        The labeled image that is a segmentation of A.
    A: (M,N) or (M,N,3) array
        The original image. Grayscale and color are supported.
        thickness : Thickness of the borders in pixels. Default is 1.
        color : Tuple of 3 uint8 RGB values.

    Returns
    -------
    (v1, v2) : tuple of (M,N,3) arrays uint8 dtype
        v1 is a colored original image. v2 is the original image with
        marked borders.
    '''

    v1 = label2rgb(L, A, bg_label=0, bg_color=(0.1,0.1,0.5), alpha=0.3, image_alpha=1)

    v1 = np.interp(v1, (0,1), (0,255)).astype(np.uint8)

    v2 = make_boundary_image(L, A)

    return (v1, v2)
