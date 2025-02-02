import numpy as np


def find_depth(center_right, center_left, baseline, focal_length_pixel):
    """
    Calculate the depth of an object using stereo vision principles.

    Args:
        center_right (tuple): The (x, y) center point of the object in the right frame.
        center_left (tuple): The (x, y) center point of the object in the left frame.
        baseline (float): The distance between the two cameras (in cm).
        focal_length_pixel (float): The focal length of the camera lens (in pixels).

    Returns:
        float: The depth (distance from the cameras) in the same unit as the baseline.
    """
    # Extract x-coordinates of the center points
    x_right = center_right[0]
    x_left = center_left[0]

    # Compute disparity (difference between the x-coordinates)
    disparity = x_left - x_right

    # Avoid division by zero
    if disparity == 0:
        return float('inf')  # Indicates an undefined depth (object is too far or cameras misaligned)

    # Calculate depth using the formula Z = (B * f) / disparity
    depth = np.abs((baseline * focal_length_pixel) / disparity)

    return depth

