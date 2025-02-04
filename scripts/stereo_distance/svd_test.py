import numpy as np

def compute_svd_depth(pts1, pts2, P_left, P_right):
    depths = []
    
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A = [
            x1 * P_left[2, :] - P_left[0, :],
            y1 * P_left[2, :] - P_left[1, :],
            x2 * P_right[2, :] - P_right[0, :],
            y2 * P_right[2, :] - P_right[1, :]
        ]
        
        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        X = V[-1]
        X = X / X[-1]  # Convert to Cartesian coordinates
        
        depths.append(X[:3])  # Store 3D coordinates
    
    return np.array(depths)

# Sample stereo camera projection matrices
f_pixel = 1066  # Example focal length in pixels
B = 12  # Baseline (cm)
cx, cy = 640, 360  # Example principal point (assuming 1280x720 image resolution)

P_left = np.array([
    [f_pixel, 0, cx, 0],
    [0, f_pixel, cy, 0],
    [0, 0, 1, 0]
])

P_right = np.array([
    [f_pixel, 0, cx, -B * f_pixel],
    [0, f_pixel, cy, 0],
    [0, 0, 1, 0]
])

# Simulated corresponding points (px) in left and right images
pts1 = np.array([[600, 350], [620, 355], [640, 360], [660, 370]])
pts2 = np.array([[590, 350], [610, 355], [630, 360], [650, 370]])

# Compute depth
depths = compute_svd_depth(pts1, pts2, P_left, P_right)

# z values are depths
print(depths)