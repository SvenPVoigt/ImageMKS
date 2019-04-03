from os import listdir, path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from random import shuffle


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
        self.files = list(i for i in self.files if i.split('.')[-1] in {'jpg', 'jpeg', 'png', 'PNG'})

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



class folder_rw(Dataset):
    def __init__(self, folderpath, prefix=None, ftype=None, filelist=True):
        super(folder_rw, self).__init__()
        if not prefix and not ftype:
            raise ValueError('Either prefix or ftype need to be defined')

        self.pre = prefix
        self.ftype = ftype
        self.path = folderpath
        self.filelist = filelist

        if filelist:
            self.files = listdir(folderpath)
            self.files = list(i for i in self.files if path.isfile(path.join(folderpath, i)))
            if ftype:
                self.files = list(i for i in self.files if i[-len(ftype):]==ftype)
            if prefix:
                self.files = list(i for i in self.files if i[:len(prefix)]==prefix)
            self.files = sorted(self.files)

    def __len__(self):
        if self.filelist:
            return len(self.files)
        else:
            return 0

    def __str__(self):
        strout = 'Is Directory? ' + str(path.isdir(self.path))

        if self.filelist:
            strout += '\n' + ',    '.join(self.files)

        return strout

    def update_list(self):
        self.files = listdir(folderpath)
        self.files = list(i for i in self.files if path.isfile(i))
        if ftype:
            self.files = list(i for i in self.files if i[-len(ftype):]==ftype)
        if prefix:
            self.files = list(i for i in self.files if i[:len(prefix)]==prefix)
        self.files = sorted(self.files)

    def __getitem__(self, idx):
        if self.filelist:
            return Image.open(self.path+self.files[idx])
        else:
            return Image.open(self.path+self.pre+'%04d'%idx+self.ftype)

    def __setitem__(self, idx, val):
        if self.ftype and self.pre:
            val.save(self.path+self.pre+'%04d'%idx+self.ftype)
        else:
            raise ValueError('Writing requires prefix and ftype')
