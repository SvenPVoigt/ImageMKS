import numpy as np
from PIL import Image
from itertools import product

cs = [0,1,2]
cs = [i for i in product(cs,cs)]

a = np.zeros((3,3,3)).astype(np.uint8)

for i in range(3):
    for j in range(3):
        c1, c2 = cs[3*i + j]
        a[i, j, c1] = 255
        a[i, j, c2] = 255

A = Image.fromarray(a)
A = A.resize((600,600))
A.show()
A.save('50-50_colors.png')
