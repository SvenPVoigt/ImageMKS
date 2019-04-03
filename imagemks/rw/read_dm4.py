import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from pycroscopy.io.translators.df_utils.dm_utils import read_dm4

print('-------------\nImported\n-------------')

import os
import struct

# basepath = '/gpfs/scratch1/0/svoigt6/In situ EXP3/Hour_00/Minute_00/Second_00/'
# filename = 'In situ EXP3_Hour_00_Minute_00_Second_00_Frame_0000.dm4'
#
# img, imgparams = read_dm4(basepath+filename)

for mn in range(5):
    file_paths = list()

    minute = 'Minute_0%d'%mn

    print(minute)

    for root, dirs, files in os.walk('/gpfs/scratch1/0/svoigt6/In situ EXP3/Hour_00/' + minute + '/', topdown=False):
        for name in sorted(files):
            file_paths.append(os.path.join(root,name))

    l = len(file_paths)

    imgs = np.memmap(minute+'.npy', dtype='float64', mode='w+', shape=(l, 512, 512))

    failed = list()

    for i, path in enumerate(sorted(file_paths)):
        j = 0
        print('%d/%d'%(i+1,l), path)
        while j < 5:
            try:
                img, imgparams = read_dm4(path)
                imgs[i,:,:] = img
                break
            except:
                print('Error')
                j += 1
                failed.append(['struct error', i, path])
        if j == 5:
            print('Complete Fail')
            failed.append(['complete fail', i, path])
        if (i+1) % 1000 == 0:
            del imgs
            imgs = np.memmap(minute+'.npy', dtype='float64', mode='r+', shape=(l, 512, 512))

    del imgs

    imgs = np.memmap(minute+'.npy', dtype='float64', mode='r+', shape=(l, 512, 512))

    print(failed)

    with open(minute + '_errors.txt', 'w+') as f:
        for err, k, path in failed:
            f.write('%s,%d,%s\n'%(err, k, path))

    print(type(imgs))

    print(imgs.shape)

    del imgs

# plt.imshow(img, cmap='gray')
# plt.savefig(filename[:-4]+'.png')
# plt.show()
