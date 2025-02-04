import numpy as np

def compute_svd_depth(pts1, pts2, P_left, P_right):
    A = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.append(x1 * P_left[2, :] - P_left[0, :]) 
        A.append(y1 * P_left[2, :] - P_left[1, :])
        A.append(x2 * P_right[2, :] - P_right[0, :])
        A.append(y2 * P_right[2, :] - P_right[1, :])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    X = X / X[-1]  # Convert homogeneous coordinates to Cartesian
    return X[:3]