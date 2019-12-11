# distutils: language=c++
#cython: cdivision=True

cimport numpy as np

cdef class graph:
  cdef int nN, nE
  cdef int* N, E

  def __init__(self):
    self.nN = 0
    self.nE = 0
    self.N = set()

  cdef from_superpixels(np.ndarray[np.float_t, ndim=2] A):
    cdef Py_ssize_t nrows = A.shape[0]
    cdef Py_ssize_t ncols = A.shape[1]
    cdef int i, j
    cdef float sum

    sum = 0

    for i in range(nrows):
      for j in range(ncols):
        sum += A[i,j]


    return sum
