import numpy as np
from imagemks.structures import weights
import matplotlib.pyplot as plt

G = weights.gauss(300, centered=True)

plt.imshow(G)
plt.show()

print('Is center max?', np.amax(G) == G[G.shape[0]//2, G.shape[1]//2])


G = weights.gauss((5, 2), theta=np.pi/4, centered=True)

plt.imshow(G)
plt.show()

print('Is center max?', np.amax(G) == G[G.shape[0]//2, G.shape[1]//2])
