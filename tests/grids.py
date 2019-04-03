import numpy as np
from imagemks.structures import grids

A = np.ones((6,3))
print('Input:\n', A)

X, Y = grids.divergent(A.shape, centered=True)

print('Centered X: \n', X)
print('Centered Y: \n', Y)
print('Center x,y:', X.shape[0]//2, X.shape[1]//2, 'with values:', X[X.shape[0]//2, X.shape[1]//2], Y[Y.shape[0]//2, Y.shape[1]//2])


X, Y = grids.divergent(A.shape, centered=False)

print('Uncentered X: \n', X)
print('Uncentered Y: \n', Y)
print('Origin x, y: 0,0 with values', X[0,0], Y[0,0])
