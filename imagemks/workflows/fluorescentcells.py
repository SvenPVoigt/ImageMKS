# Importing general libraries
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings("ignore")

# Importing specific functions
from skimage.morphology import remove_small_objects, remove_small_holes, watershed
from skimage.feature import corner_peaks
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops
from skimage.color import label2rgb

# Importing ImageMKS functions
from ..filters import fftgauss, local_avg, smooth_binary
from ..structures import donut
from ..masking import maskfourier
from ..visualization import make_boundary_image
from ..rw import dirload



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
        Either muscle_progenitor or bone_stem. More support coming soon.

    Returns
    -------
    params : dictionary
        Params defines smooth_size, intensity_curve, short_th_radius, long_th_radius,
        min_frequency_to_remove, max_frequency_to_remove,
        max_size_of_small_objects_to_remove, power_adjust, peak_min_distance,
        size_after_watershed_to_remove, and cyto_local_avg_size.
    '''
    if cell_type is 'muscle_progenitor':
        params = {
            'smooth_size': 3,
            'intensity_curve': 2,
            'short_th_radius': 50,
            'long_th_radius': 600,
            'max_size_of_small_objects_to_remove': 300,
            'power_adjust': 1,
            'peak_min_distance': 10,
            'size_after_watershed_to_remove': 300,
            'cyto_local_avg_size': 200
            }

        return params

    elif cell_type is 'bone_stem':

        params = {
            'smooth_size': 3,
            'intensity_curve': 3,
            'short_th_radius': 100,
            'long_th_radius': 800,
            'max_size_of_small_objects_to_remove': 1100,
            'power_adjust': 1,
            'peak_min_distance': 10,
            'size_after_watershed_to_remove': 1100,
            'cyto_local_avg_size': 200
            }

        return params

    else:
        print('Sorry this cell type is not yet supported.')


def all_measurements():
    measures = ['Cell_Index', 'Nuc_Area_um2', 'Nuc_Perimeter_um', 'Nuc_Area_Factor',
                'Nuc_Major_L_um', 'Nuc_Minor_L_um', 'Nuc_eccentricity', 'Nuc_orientation',
                'Nucleus_eq_diameter_um', 'Cyto_Area_um2', 'Cyto_Perimeter_um', 'Cyto_Area_Factor',
                'Cyto_orientation', 'Cyto_Major_L_um', 'Cyto_Minor_L_um']

    return measures


def segment_fluor_cells(N, C, smooth_size, intensity_curve, short_th_radius, long_th_radius, max_size_of_small_objects_to_remove, peak_min_distance, size_after_watershed_to_remove, cyto_local_avg_size, pix_size):
    '''
    Segments fluorescent cells.

    Parameters
    ----------
    N : (M,N,3) numpy array
        A color image of nuclei with size (M,N,3)
    C : (M,N,3) numpy array
        A color image of the cytoskeleton with same size as the nucleus image (M,N,3).
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
    pix_size : float
        Real measurement in micrometers of a pixel side.

    Returns
    -------
    (N, C) : list of (M,N) numpy arrays. Long dtype
        N is a labeled nucleus image. Where each label corresponds to an individual
        cell. 0 corresponds to the background. C is a labeled cytockeleton image. The
        labels correspond to the closest nucleus in N.
    '''

    # The parameters are adjusted based on an image with pixel resolution of 0.48
    zoomLev = 0.48 / pix_size

    N = np.sum(np.array(N), axis=2)
    N = ( (( N-np.amin(N)) / np.ptp(N)) )

    C = np.sum(np.array(C), axis=2)
    C = ( (( C-np.amin(C)) / np.ptp(C)) )

    # Step 1: smoothing intensity values and smoothing out peaks
    N = fftgauss(N, smooth_size, pad_type='edge')

    # Step 2: contrast enhancement by scaling intensities (from 0-1) on a curve
    ########  many other methods can be implemented for this step which could benefit the segmentation
    N = np.power(N/np.amax(N), intensity_curve)

    # Step 3: short and long range local avg threshold
    th_short = N > local_avg(N, short_th_radius)
    th_long = N > local_avg(N, long_th_radius)

    th_N = (th_short*th_long)
    del th_short, th_long

    # Step 4: remove small objects
    th_N = remove_small_objects(th_N, 20)
    th_N = remove_small_objects(th_N, max_size_of_small_objects_to_remove * (zoomLev))

    # Step 5: distance transform and generate markers from peaks
    distance = ndi.distance_transform_edt(th_N)
    peak_markers = corner_peaks(distance, min_distance=peak_min_distance, indices=False)
    peak_markers = ndi.label(peak_markers)[0]

    # Step 6: separate touching nuclei using the watershed markers
    label_N = watershed(th_N, peak_markers, mask=th_N)

    # Step 7: removing small regions after the watershed segmenation
    label_N = remove_small_objects(label_N, size_after_watershed_to_remove * (zoomLev))

    # Step 8: reassigning labels, so that they are continuously numbered
    old_labels = np.unique(label_N)
    for i in range(len(old_labels)):
        label_N[label_N == old_labels[i]] = i

    # Step 14: local threshold of the cytoskeleton
    label_C = C > local_avg(C, cyto_local_avg_size)
    label_C = smooth_binary(label_C, add_cond=0.5)

    # Step 15: generate relabeled markers from the nuclei centroids
    new_markers = _gen_marks(label_N)

    # Step 16: watershed of cytoskeleton using new_markers
    label_C = watershed(label_C, new_markers, mask=label_C.astype(np.bool_))

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
        Cell_Index, Nuc_Area_um2, Nuc_Perimeter_um, Nuc_Area_Factor,
        Nuc_Major_L_um, Nuc_Minor_L_um, Nuc_eccentricity, Nuc_orientation,
        Nucleus_eq_diameter_um, Cyto_Area_um2, Cyto_Perimeter_um, Cyto_Area_Factor,
        Cyto_orientation, Cyto_Major_L_um, Cyto_Minor_L_um
    '''

    nuc_props = regionprops(label_Nuc)

    cell_index = 1

    col_names = all_measurements()

    prop_df = pd.DataFrame(columns = col_names)

    for i in range(len(nuc_props)):
        nucleus_A = nuc_props[i].area / (pix_size**2)
        nucleus_P = nuc_props[i].perimeter / pix_size
        nucleus_SF = nucleus_A / (nucleus_P**2)

        cyto_props = regionprops((label_Cyto==i).astype(np.int))

        try:
            cyto_A = cyto_props[0].area  / (pix_size**2)
            cyto_P = cyto_props[0].perimeter  / pix_size
            cyto_SF = cyto_A / (cyto_P**2)
            cyto_orientation = cyto_props[0].orientation
            cyto_Major_L_um = cyto_props[0].major_axis_length
            cyto_Minor_L_um = cyto_props[0].minor_axis_length

        except:
            cyto_A = None
            cyto_P = None
            cyto_SF = None
            cyto_orientation = None
            cyto_Major_L_um = None
            cyto_Minor_L_um = None

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
                       'Cyto_orientation': (cyto_orientation,),
                       'Cyto_Major_L_um': (cyto_Major_L_um,),
                       'Cyto_Minor_L_um': (cyto_Minor_L_um,)}))
        cell_index += 1

    # Reordering the columns
    prop_df = prop_df[col_names]


    return prop_df


def visualize_fluor_cells(L, A, thickness=1, bg_color='b', engine='matplotlib', figsize=(10,10)):
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
    thickness: int
        Thickness of the yellow borders drawn on the im

    Returns
    -------
    (v1, v2) : tuple of (M,N,3) arrays uint8 dtype
        v1 is a colored original image. v2 is the original image with
        marked borders.
    '''

    A = np.array(A)

    if bg_color is 'b':
        bg_color=(0.1,0.1,0.5)
    elif bg_color is 'g':
        bg_color=(0.1,0.5,0.1)

    A = label2rgb(L, A, bg_label=0, bg_color=bg_color, alpha=0.1, image_alpha=1)

    A = np.interp(A, (0,1), (0,255)).astype(np.uint8)

    A = make_boundary_image(L, A, thickness=thickness)

    if engine == 'matplotlib':
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.imshow(A)
        plt.show(fig)
    elif engine == 'PIL':
        A = Image.fromarray(A)
        A.show()
    elif engine == 'getimg':
        A = Image.fromarray(A)
        return A


def getparams(path):
    p = default_parameters('muscle_progenitor')

    with open(path, 'w+') as f:
        json.dump(p, f)


def segment(path_n, path_c, save_n, save_c, path_p, pxsize):
    print('Loading nuclei from `%s` and saving labels to `%s`.'%(path_n, save_n))
    nucleus_loader = dirload(path_n)
    print('Loading cytoskeletons from `%s` and saving labels to `%s`.'%(path_c, save_c))
    cyto_loader = dirload(path_c)
    print('Loading parameters from `%s`.'%path_p)
    with open(path_p, 'r') as f:
        p = json.load(f)

    for i, (N, C) in enumerate(zip(nucleus_loader, cyto_loader)):
        # Make sure that the image ids for nucleus and cytoskeleton are the same
        assert nucleus_loader.getname(i) == cyto_loader.getname(i)

        print('\nSegmenting Nucleus and Cytoskeleton with id: `%s`.'%nucleus_loader.getname(i))

        # Do segmentation. Pass all of the parameters in order.
        Label_N, Label_C = segment_fluor_cells(N, C, p['smooth_size'], p['intensity_curve'], p['short_th_radius'], p['long_th_radius'], p['max_size_of_small_objects_to_remove'], p['peak_min_distance'], p['size_after_watershed_to_remove'], p['cyto_local_avg_size'], pxsize)

        # Convert Images to PIL objects
        Label_N = visualize_fluor_cells(Label_N, N, thickness=2, bg_color='b', engine='getimg')
        Label_C = visualize_fluor_cells(Label_C, C, thickness=2, bg_color='g', engine='getimg')

        # Save Labeled Images to File
        path = os.path.join(save_n, nucleus_loader.getname(i) + '.png')
        Label_N.save(path)
        print('Saved labeled nuclei to `%s`.'%path)

        path = os.path.join(save_c, nucleus_loader.getname(i) + '.png')
        Label_C.save(path)
        print('Saved labeled cytoskeletons to `%s`.'%path)



def measure(path_n, path_c, save_m, path_p, pxsize):
        print('Loading nuclei from `%s`.'%path_n)
        nucleus_loader = dirload(path_n)
        print('Loading cytoskeletons from `%s`.'%path_c)
        cyto_loader = dirload(path_c)
        print('Loading parameters from `%s`.'%path_p)
        with open(path_p, 'r') as f:
            p = json.load(f)
        print('Saving measurements to `%s`.'%save_m)

        # Initialize Dataframe to save measurements
        col_names = ['Image_ID',] + all_measurements()
        M = pd.DataFrame(columns = col_names)

        for i, (N, C) in enumerate(zip(nucleus_loader, cyto_loader)):
            # Make sure that the image ids for nucleus and cytoskeleton are the same
            assert nucleus_loader.getname(i) == cyto_loader.getname(i)

            print('\nAnalyzing Nucleus and Cytoskeleton with id: `%s`.'%nucleus_loader.getname(i))

            # Do segmentation. Pass all of the parameters in order.
            Label_N, Label_C = segment_fluor_cells(N, C, p['smooth_size'], p['intensity_curve'], p['short_th_radius'], p['long_th_radius'], p['max_size_of_small_objects_to_remove'], p['peak_min_distance'], p['size_after_watershed_to_remove'], p['cyto_local_avg_size'], pxsize)

            # Get the measurements
            new_M = measure_fluor_cells(Label_N, Label_C, pxsize)
            new_M.insert(0, 'Image_ID', nucleus_loader.getname(i))

            # Append the measurements to the larger dataframe
            M = M.append(new_M)

        # Save Labeled Images to File
        M.to_csv(save_m)
        print('Saved measurements to `%s`.'%save_m)
