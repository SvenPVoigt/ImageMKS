import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import relabel_sequential
from skimage.morphology import watershed


def segment_cells(B, G, zoom, p):
    '''
    B is the blue channel image that corresponds to the nuclei.
    G is the green channel image that corresponds to the cytoskeleton.
    zoom is the magnification used to take the image.
    p is a dictionary of parameters used for the segmentation.
    '''
    smooth_size, intensity_curve, short_th_radius, long_th_radius, min_frequency_to_remove,
                  max_frequency_to_remove, max_size_of_small_objects_to_remove, peak_min_distance,
                  size_after_watershed_to_remove, cyto_local_avg_size, zoomLev = p

    # Step 1: smoothing intensity values and smoothing out peaks
    B = gBlur(B, p['smooth_size'])

    # Step 2: contrast enhancement by scaling intensities (from 0-1) on a curve
    ########  many other methods can be implemented for this step which could benefit the segmentation
    B = np.power(B/np.amax(B), p['intensity_exponent'])

    # Step 3: short range local avg threshold
    th_short = B > local_avg(B, p['short_th_radius'])

    # Step 4: long range local avg threshold
    th_long = B > local_avg(B, p['long_th_radius'])

    # Step 5: long && short
    th_B = (th_short*th_long)
    del th_short, th_long

    # Step 6: remove the short and long frequencies
    # freq_B = remove_freq(th_B, cutmax = max_frequency_to_remove, cutmin = min_frequency_to_remove)

    # Step 7: threshold the inverse fourier transform
    # th_m = th_freq_1(freq_B)
    # th_B = freq_B > th_m
    # del freq_B

    # Step 8: remove small objects
    size_rem = p['max_size_of_small_objects_to_remove'] * (zoom**p['power_adjust'])
    th_B = do_rem_obj(th_B, size_rem, 'less')

    # Step 10: mark the maxima in the distance transform and assign labels
    peak_markers = feature.corner_peaks(ndi.distance_transform_edt(th_B),
                                        min_distance=p['peak_min_distance'],
                                        indices=False)
    peak_markers = ndi.label(peak_markers)[0]

    # Step 11: separate touching nuclei using the watershed markers
    label_B = watershed(th_B, peak_markers, mask=th_B)

    # Step 12: removing small regions after the watershed segmenation
    size_rem = p['size_after_watershed_to_remove'] * (zoom**p['power_adjust'])

    label_B = rem_obj(label_B > 0, size_rem, gre_or_less='less')

    # Step 13: reassigning labels, so that they are continuously numbered
    # old_labels = np.unique(label_B)
    # for i in range(len(old_labels)):
    #     label_B[label_B == old_labels[i]] = i
    label_B = relabel_sequential(label_B)

    # Step 14: local threshold of the cytoskeleton
    th_G = G > local_avg(G, p['cyto_local_avg_size'])

    # Step 15: generate relabeled markers from the nuclei centroids
    new_markers = gen_marks(label_B)

    # Step 16: watershed of cytoskeleton using new_markers
    label_G = watershed(th_G, new_markers, mask=th_G.astype(np.bool_))

    return (label_B, label_G)
