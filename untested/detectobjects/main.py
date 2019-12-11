import numpy as np
from graph import graph

# Import the image


# Generate the superpixels
A = np.arange(15).reshape((3, 5))


# Generate the graph
G = graph()
G.from_matrix(A)
