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


def _th_max_hist(img, num_bins=100):
    '''
    Method for thresholding at the max of a histogram.

    Parameters
    ----------
    img : An image of size (M,N).
    num_bins : Number of bins used for histogram. Optional.

    Returns
    -------
    A threshold value.
    '''

    minbin = np.amin(img)
    maxbin = np.amax(img)
    bins = np.linspace(minbin, maxbin, num_bins)
    hist, bins = np.histogram(img.ravel(), bins = bins)

    bins_mid = (bins[1:] + bins[:-1])/2
    big_peak = bins_mid[np.argmax(hist)]
    th = 0.5 + big_peak

    return th


def segment_fluor_cells(imgNuc, imgCyto, smooth_size, intensity_curve, short_th_radius,
                long_th_radius, min_frequency_to_remove, max_frequency_to_remove,
                max_size_of_small_objects_to_remove, peak_min_distance,
                size_after_watershed_to_remove, cyto_local_avg_size, zoomLev):
    '''
    Segments fluorescent cells.

    Parameters
    ----------
    imgNuc : An image of nuclei with size (M,N)
    imgCyto : An image of the cytoskeleton with same size (M,N).
    smooth_size : The sigma of the gaussian.
    intensity_curve : Exponent of the curve used to fit intensities on range [0,1]
    short_th_radius : Radius of neighborhood used to calculate a local
        average threshold.
    long_th_radius : Radius of neighborhood used to calculate a local
        average threshold
    min_frequency_to_remove : Frequency in pixels used to define donut mask.
    max_frequency_to_remove : Frequency in pixels used to define donut mask.
    max_size_of_small_objects_to_remove : Size beneath which no cells can exist.
    peak_min_distance : Min distance between nuclei.
    size_after_watershed_to_remove : Size beneath which no cells can exist.
        Calculated after watershed.
    cyto_local_avg_size : Radius of neighborhood used to calculate a local
        average threshold
    zoomLev : Real magnification of the image.

    Returns
    -------
    (N, C) : N is a labeled nucleus image. Where each label corresponds to an individual
        cell. 0 corresponds to the background. C is a labeled cytockeleton image. The
        labels correspond to the closest nucleus in N.
    '''

    # Step 1: smoothing intensity values and smoothing out peaks
    imgNuc = fftgauss(imgNuc, smooth_size)

    # Step 2: contrast enhancement by scaling intensities (from 0-1) on a curve
    ########  many other methods can be implemented for this step which could benefit the segmentation
    imgNuc = np.power(imgNuc/np.amax(imgNuc), intensity_curve)

    # Step 3: short range local avg threshold
    th_short = imgNuc > local_avg(imgNuc, short_th_radius)[0]

    # Step 4: long range local avg threshold
    th_long = imgNuc > local_avg(imgNuc, long_th_radius)[0]

    # Step 5: long && short
    th_Nuc = (th_short*th_long)
    del th_short, th_long

    # Step 6: remove the short and long frequencies
    mask = donut(r_outer=max_frequency_to_remove, r_inner=min_frequency_to_remove)
    freq_Nuc = maskfourier(th_Nuc, mask)

    # Step 7: threshold the inverse fourier transform
    th_m = _th_freq_1(freq_Nuc)
    th_Nuc = freq_Nuc > th_m
    del freq_Nuc

    # Step 8: remove small objects
    th_Nuc = remove_small_objects(th_Nuc, max_size_of_small_objects_to_remove * (zoomLev))

    # Step 9: distance transform
    distance = ndi.distance_transform_edt(th_Nuc)

    # Step 10: mark the maxima in the distance transform and assign labels
    peak_markers = corner_peaks(distance, min_distance=peak_min_distance, indices=False)
    peak_markers = ndi.label(peak_markers)[0]

    # Step 11: separate touching nuclei using the watershed markers
    label_Nuc = watershed(th_Nuc, peak_markers, mask=th_Nuc)

    # Step 12: removing small regions after the watershed segmenation
    sizes_matrix = ndi.sum(label_Nuc>0, label_Nuc, np.unique(label_Nuc))
    rem_region = sizes_matrix < ( size_after_watershed_to_remove * (zoomLev) )
    remove_pixel = rem_region[label_Nuc]
    label_Nuc[remove_pixel] = 0

    # Step 13: reassigning labels, so that they are continuously numbered
    old_labels = np.unique(label_Nuc)
    for i in range(len(old_labels)):
        label_Nuc[label_Nuc == old_labels[i]] = i

    # Step 14: local threshold of the cytoskeleton
    th_Cyto = imgCyto > local_avg(imgCyto, cyto_local_avg_size)[0]

    # Step 15: generate relabeled markers from the nuclei centroids
    new_markers = relabel_sequential(label_Nuc)

    # Step 16: watershed of cytoskeleton using new_markers
    label_Cyto = watershed(th_Cyto, new_markers, mask=th_Cyto.astype(np.bool_))

    return [label_Nuc, label_Cyto]


def measure_fluor_cells(label_Nuc, label_Cyto):
    '''
    Generates measurements for labeled Nucleus images and labeled Cytoskeleton images.

    Parameters
    ----------
    label_Nuc : A labeled nucleus image. Where each label corresponds to an individual
        cell. 0 corresponds to the background.
    label_Cyto : A labeled cytockeleton image. The labels correspond to the closest
        nucleus in N. 0 corresponds to the background.

    Returns
    -------
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

        prop_df = prop_df.append(pan.DataFrame(
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
    L : The labeled image that is a segmentation of A.
    A : The original image. Grayscale and color are supported.
        thickness : Thickness of the borders in pixels. Default is 1.
        color : Tuple of 3 uint8 RGB values.

    Returns
    -------
    (v1, v2) : v1 is a colored original image. v2 is the original image with
        marked borders.
    '''

    v1 = label2rgb(L, A, bg_label=0, bg_color=(0.1,0.1,0.5), alpha=0.3, image_alpha=1)

    v1 = np.interp(v1, (0,1), (0,255)).astype(np.uint8)

    v2 = make_boundary_image(L, A)

    return (v1, v2)
