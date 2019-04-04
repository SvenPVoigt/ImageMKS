import numpy as np
from imagemks.structures import weights, grids
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

G = weights.gauss(300, centered=True)

plt.imshow(G)
plt.show()

print('Is center max?', np.amax(G) == G[G.shape[0]//2, G.shape[1]//2])


G = weights.gauss((5, 2), theta=np.pi/4, centered=True)

plt.imshow(G)
plt.show()

print('Is center max?', np.amax(G) == G[G.shape[0]//2, G.shape[1]//2])


def plot_3d(A):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = grids.divergent(A.shape)

    surf = ax.plot_trisurf(X.ravel(), Y.ravel(), A.ravel(), cmap=plt.cm.Blues)
    plt.show()


cone = weights.conical(50)

plot_3d(cone)

cone = weights.conical(5)

plot_3d(cone)

drop = weights.drop(50, 0.4)

plot_3d(drop)
