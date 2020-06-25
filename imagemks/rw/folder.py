import os
from os import listdir, path
from PIL import Image
from random import shuffle
from scipy.io import loadmat, savemat
import numpy as np


class dirload():
    '''
    Reads a sorted list of files from the specified directory if those files are of type tif, tiff, jpg, jpeg, png, or PNG. Can specify the type if desired.

    Will then allow accessing these files via slicing.

    Parameters
    ----------
    folderpath : string
    order : string, optional
        Any of {sorted or random}.
    mode : string, optional
        Any of {train, test, predict, read}.
    train_T : function, optional
        Function that transforms the training input.
    test_T : function, optional
        Function that transforms the test input.
    predict_T : function, optional
        Function that transforms the predict input.
    read_T : function, optional
        Function that transforms the read input.
    ftype : str, optional
        Specifies the type of the file that is includes in files
    prefix : str, optional
        Specifies the prefix of the file that is included in files

    Attributes
    ----------
    path : string
        Folder path.
    mode : string
        Specifies how the reader class loads data.
    files : list
        List of all files in folder.
    T : function
        Function that transforms loaded data.
    ftype : str, optional
        Specifies the type of the file that is includes in files
    prefix : str, optional
        Specifies the prefix of the file that is included in files
    '''


    def __init__(self, folderpath, order='sorted', mode='predict', train_T=lambda x:x, test_T=lambda x:x, predict_T=lambda x:x, read_T=lambda x:x, ftype=None, prefix=None):
        assert order in {'sorted', 'random'}
        assert mode in {'train', 'test', 'predict', 'read'}
        self.T = {'train':train_T, 'test':test_T, 'predict':predict_T, 'read':read_T}

        self.path = folderpath
        self.mode = mode

        self.ftype = ftype
        self.prefix = prefix

        self.files = listdir(folderpath)
        self.files = list(i for i in self.files if path.isfile(path.join(folderpath, i)))

        if ftype:
            self.files = list(i for i in self.files if i[-len(ftype):]==ftype)
        else:
            self.files = list(i for i in self.files if i.split('.')[-1] in {'jpg', 'jpeg', 'png', 'PNG', 'tif', 'tiff'})

        if prefix:
            self.files = list(i for i in self.files if i[:len(self.prefix)]==self.prefix)

        if order is 'sorted':
            self.files = sorted(self.files)
        elif order is 'random':
            self.files = shuffle(self.files)


    def __len__(self):
        return len(self.files)


    def __str__(self):
        return 'Is %s a directory? '%self.path + str(path.isdir(self.path))


    def print_all_files_(self):
        '''
        Function that prints all files in the folder.

        Prints
        ------
        all_files : str
            Each file separated by `,` in files.
        '''
        return '\n' + ',    '.join(self.files)


    def change_mode(self, mode):
        '''
        Function that changes how the reader loads data.

        Parameters
        ----------
        mode : str
        '''
        assert mode in {'train', 'predict', 'read'}
        self.mode = mode


    def update_list(self):
        '''
        If any files were added to or removed from the folder, then this
        function should be run to update files that will be loaded.

        Updates
        -------
        files : str
            All files in the folder that meet the prefix and ftype rules.
        '''
        self.files = listdir(folderpath)
        self.files = list(i for i in self.files if path.isfile(i))
        if ftype:
            self.files = list(i for i in self.files if i[-len(self.ftype):]==self.ftype)
        if prefix:
            self.files = list(i for i in self.files if i[:len(self.prefix)]==self.prefix)
        self.files = sorted(self.files)


    def getpath(self, idx):
        return path.join(self.path, self.files[idx])


    def getname(self, idx):
        return ''.join(self.files[idx].split('.')[:-1])


    def __getitem__(self, idx):
        return self.T[self.mode](Image.open(path.join(self.path, self.files[idx])))


class rwformat():
    '''
    Reads and writes from a folder by combining the prefix, ftype,
    and passed idx value. Can set width of the idx with idx_len.

    Parameters
    ----------
    folderpath : string
    idx_len : int, optional
        How long the numeric value of the name should be.
    ftype : str, optional
        Specifies the type of the file that is includes in files
    prefix : str, optional
        Specifies the prefix of the file that is included in files

    Attributes
    ----------
    path : string
        Folder path.
    idx_len : int, optional
        How long the numeric value of the name should be.
    ftype : str, optional
        Specifies the type of the file that is includes in files
    pre : str, optional
        Specifies the prefix of the file that is included in files
    '''
    def __init__(self, folderpath, idx_len=None, prefix='', ftype='.png'):

        self.path = folderpath
        self.idx_len = idx_len
        self.pre = prefix
        self.ftype = ftype

    def __len__(self):
        '''
        Finds how many files have been written to the folder or are
        currently in the folder
        '''
        files = listdir(self.path)
        files = list(i for i in files if path.isfile(path.join(self.path, i)))
        if self.ftype:
            files = list(i for i in files if i[-len(self.ftype):]==self.ftype)
        if self.pre:
            files = list(i for i in files if i[:len(self.pre)]==self.pre)
        return len(files)

    def __str__(self):
        return 'Is %s a directory? '%self.path + str(path.isdir(self.path))

    def print_all_files_(self):
        '''
        Function that prints all files in the folder.

        Prints
        ------
        all_files : str
            Each file separated by `,` in files.
        '''
        return '\n' + ',    '.join(self.files)

    def __getitem__(self, idx):
        if self.idx_len:
            num = '{number:0{width}d}'.format(width=self.idx_len, number=idx)
        else:
            num = str(idx)

        path = os.path.join(self.path, self.pre + num + self.ftype)

        if self.ftype in {'.jpeg', '.jpg', '.png', '.PNG', '.tif', '.tiff'}:
            return Image.open(path)
        elif self.ftype == '.mat':
            return loadmat(path)
        elif self.ftype == '.npy':
            return np.load(path)
        else:
            raise ValueError('Unsupported filetype (ftype)')

    def __setitem__(self, idx, val):
        if self.idx_len:
            num = '{number:0{width}d}'.format(width=self.idx_len, number=idx)
        else:
            num = str(idx)

        if self.ftype in {'.jpeg', '.jpg', '.png', '.PNG', '.tif'}:
            val.save(self.path+self.pre+num+self.ftype)
        elif self.ftype == '.mat':
            savemat(self.path+self.pre+num+self.ftype, val)
        elif self.ftype == '.npy':
            np.save(self.path+self.pre+num+self.ftype, val)
