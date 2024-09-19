import numpy as np

def identity_like_matrix(n, m):
    # Initialize an n by m matrix with zeros
    matrix = np.zeros((n, m))
    # Fill the diagonal with 1's
    for i in range(min(n, m)):
        matrix[i, i] = 1
    return matrix