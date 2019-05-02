from os import listdir, path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from scipy.io import loadmat, savemat
import numpy as np


class readlist(Dataset):
    ''' Reads a sorted list of files from the specified directory if those files are of type jpg, jpeg, png, or PNG. '''


    def __init__(self, folderpath, order='sorted', mode='predict', train_T=lambda x:x, test_T=lambda x:x, predict_T=lambda x:x, read_T=lambda x:x):
        assert order in {'sorted', 'random'}
        assert mode in {'train', 'test', 'predict', 'read'}
        self.T = {'train':train_T, 'test':test_T, 'predict':predict_T, 'read':read_T}

        super(readlist, self).__init__()

        self.path = folderpath
        self.mode = mode

        self.files = listdir(folderpath)
        self.files = list(i for i in self.files if path.isfile(path.join(folderpath, i)))
        self.files = list(i for i in self.files if i.split('.')[-1] in {'jpg', 'jpeg', 'png', 'PNG', 'tif'})

        if order is 'sorted':
            self.files = sorted(self.files)
        elif order is 'random':
            self.files = shuffle(self.files)


    def __len__(self):
        return len(self.files)


    def __str__(self):
        return 'Is %s a directory? '%self.path + str(path.isdir(self.path))


    def print_all_files_(self):
        return '\n' + ',    '.join(self.files)


    def update_list(self):
        self.files = listdir(folderpath)
        self.files = list(i for i in self.files if path.isfile(i))
        if ftype:
            self.files = list(i for i in self.files if i[-len(ftype):]==ftype)
        if prefix:
            self.files = list(i for i in self.files if i[:len(prefix)]==prefix)
        self.files = sorted(self.files)


    def __getitem__(self, idx):
        return self.T[self.mode](Image.open(self.path+self.files[idx]))


class writeindex(Dataset):
    def __init__(self, folderpath, order=None, prefix=None, ftype=None):
        super(rwtype, self).__init__()

        assert all(isinstance(prefix, str), isinstance(ftype,str)), 'prefix and ftype need to be defined for w mode.'

        self.pre = prefix

        self.ftype = ftype

        self.path = folderpath
        self.order = order

    def __len__(self):
        self.files = listdir(folderpath)
        self.files = list(i for i in self.files if path.isfile(path.join(folderpath, i)))
        if ftype:
            self.files = list(i for i in self.files if i[-len(ftype):]==ftype)
        if prefix:
            self.files = list(i for i in self.files if i[:len(prefix)]==prefix)
        return len(self.files)

    def __str__(self):
        return 'Is %s a directory? '%self.path + str(path.isdir(self.path))

    def print_all_files_(self):
        return '\n' + ',    '.join(self.files)

    def __getitem__(self, idx):
        if self.order:
            num = '{number:0{width}d}'.format(width=self.order, number=idx)
        else:
            num = str(idx)

        return Image.open(self.path+self.pre+num+self.ftype)

    def __setitem__(self, idx, val):
        if self.order:
            num = '{number:0{width}d}'.format(width=self.order, number=idx)
        else:
            num = str(idx)

        if ftype in {'.jpeg', '.jpg', '.png', '.PNG', '.tif'}:
            val.save(self.path+self.pre+num+self.ftype)
        elif ftype == '.mat':
            savemat(self.path+self.pre+num+self.ftype, val)
        elif ftype == '.npy':
            np.save(self.path+self.pre+num+self.ftype, val)
