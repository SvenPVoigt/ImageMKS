import numpy as np
from imagemks.structures import shapes

print(shapes.circle(2, (5,8)))
for i in (0.2, 0.8, 1, 1.5, 2, 2.5, 3, 4):
    print('Radius=%.2f:\n'%i, shapes.circle(i))
