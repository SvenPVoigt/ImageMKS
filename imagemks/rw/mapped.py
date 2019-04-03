import numpy as np

# def init_stacks(shape, m=['r', 'w+', 'w+']):
#     TEM_Stack = np.memmap('%s.npy'%minute, dtype='float64', mode=m[0], shape=shape)
#     Corrected_Stack = np.memmap(path_scratch + 'Corrected_Stack_%s.npy'%minute, dtype='float64', mode=m[1], shape=shape)
#     Segmented_Stack = np.memmap(path_scratch + 'Segmented_Stack_%s.npy'%minute, dtype='bool', mode=m[2], shape=shape)
#
#     return (TEM_Stack, Corrected_Stack, Segmented_Stack)
#
#
# path_scratch = '/gpfs/scratch1/0/svoigt6/segmented_TEM/'


class mapped_rw():
    def __init__(self, fname, path, shape, dtype='float32'):
        self.fname = fname
        self.path = path
        self.dtype = dtype
        self.shape = shape

    def __len__(self):
        return shape[0]

    def createfile(self):
        self.map = np.memmap(self.path+self.fname, dtype=self.dtype, mode='w+', shape=self.shape)
        del self.map

    def load(self, mode='r+'):
        self.mode = mode
        self.map = np.memmap(self.path+self.fname, dtype=self.dtype, mode=self.mode, shape=self.shape)

    def unload(self):
        del self.map

    def flush(self):
        del self.map
        self.map = np.memmap(self.path+self.fname, dtype=self.dtype, mode=self.mode, shape=self.shape)

    def __getitem__(self, idx):
        if self.mode == 'r':
            return self.map[idx, shape[1], shape[2]]
        else:
            print('Reading not supported for the selected mode. Please unload and load in mode=r.')

    def __setitem__(self, idx, val):
        if self.mode == 'r+':
            self.map[idx, :, :] = val
        else:
            print('Writing not supported for the selected mode. Please unload and load in mode=r+. May need to run createfile.')
