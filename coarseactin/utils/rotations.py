import numpy as np
import sklearn

# Random rotation matrix
def random_rotation():
    """Generate a 3D random rotation matrix.
    Returns:
        np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M


# Optimal rotation matrix
# The longest coordinate is X, then Y, then Z.
def optimal_rotation(coords):
    c = coords.copy()
    c -= c.mean(axis=0)
    pca = sklearn.decomposition.PCA()
    pca.fit(c)
    # Change rotoinversion matrices to rotation matrices
    rot = pca.components_[[0, 1, 2]]
    if np.linalg.det(rot) < 0:
        rot = -rot
        #print(rot, np.linalg.det(rot))
    return rot