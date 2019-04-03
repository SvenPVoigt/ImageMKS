import numpy as np
#from numba import jit

def location_grid(stats_shape, step):

    axes = tuple()

    for i in stats_shape:
        bot = -(i//2 * step)
        ## The +1 -1 is to handle even cases. The center point of stats gets shifted to the shape[i]//2
        ## position. Therefore, stats with even shapes have a larger (-) vector and smaller (+) vector.
        top = ((i + 1) // 2 - 1) * step
        axes += (np.linspace(bot, top, i),)

    if len(stats_shape) == 2:
        return np.meshgrid(axes[0], axes[1], indexing='ij')

    if len(stats_shape) == 3:
        return np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')


#@jit(nopython=True)
def _do_aggregate_sum(arr, ind_to_agg_by, x):
    ''' x has to be all zeros with length of the ind_to_agg_by '''
    for i in range(0, len(ind_to_agg_by)-1):
        a = ind_to_agg_by[i]
        b = ind_to_agg_by[i+1]
        for j in range(a, b):
            x[i] += arr[j]
    return x


def aggregate_sum(arr, ind_to_agg_by):
    # Initialize array to speed up jit function
    x = np.zeros( len(ind_to_agg_by) )
    return _do_aggregate_sum(arr, ind_to_agg_by, x)


def stats_to_rdf(stats_norm=None, stats_not_norm=None, mask=None, step=None):

    if stats_norm is not None and stats_not_norm is not None:
        raise valueError('May not enter more than one statistic array!')
    if stats_norm is None and stats_not_norm is None:
        raise ValueError('Need to input at least one statistic array!')
    if stats_not_norm is not None and mask is None:
        raise ValueError('Mask needs to be supplied for unnormalized statistic array!')
    if step is None:
        step = 1

    if stats_norm is not None:
        D = location_grid(stats_norm.shape, step=step)
        if len(D) == 2:
            D = np.sqrt(D[0]**2 + D[1]**2)
        elif len(D) == 3:
            D = np.sqrt(D[0]**2 + D[1]**2 + D[2]**2)
        D = D.ravel()
        ind = np.argsort(D)
        D = D[ind]
        stats_norm = stats_norm.ravel()[ind]

        r, ind_agg, norm_sum = np.unique(D, return_index=True, return_counts=True)

        prob_sum = aggregate_sum(stats_norm, ind_agg)

        rdf = (r, np.divide(prob_sum, norm_sum, where=norm_sum!=0))

        return rdf

    elif stats_not_norm is not None:
        D = location_grid(stats_not_norm.shape, step=step)
        D = np.sqrt(D[0]**2 + D[1]**2 + D[2]**2)
        D = D.ravel()
        ind = np.argsort(D)
        D = D[ind]
        stats_not_norm = stats_not_norm.ravel()[ind]
        mask = mask.ravel()[ind]

        r, ind_agg = np.unique(D, return_index=True)

        prob_sum = aggregate_sum(stats_not_norm, ind_agg)
        norm_sum = aggregate_sum(mask, ind_agg)

        rdf = (r, np.divide(prob_sum, norm_sum, where=norm_sum!=0))

        return rdf
