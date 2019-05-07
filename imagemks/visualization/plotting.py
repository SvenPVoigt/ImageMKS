import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def histvis(a, nbins, col1='blue', col2='cyan', label='x', skip_ticks=1, figsize=(10,6), savepath=None):
    hist, bins = np.histogram(a.ravel(), bins=nbins)

    colors = [col1 if i%2==0 else col2 for i in range(nbins)]

    fig, ax = plt.subplots(1,1,figsize=figsize)

    rcParams['axes.labelpad'] = 15
    rcParams['figure.autolayout'] = 1

    ax.bar(bins[:-1], hist, width=(bins[1]-bins[0]), color=colors, align='edge')
    ax.set_xlabel(label, fontsize=22)
    ax.set_ylabel('Count', fontsize=22)
    ax.set_xticks(ticks=bins[::skip_ticks])
    ax.set_xticklabels(labels=np.around(bins[::skip_ticks], 2))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    if savepath:
        fig.savefig(savepath, dpi=300)

    plt.show(fig)
