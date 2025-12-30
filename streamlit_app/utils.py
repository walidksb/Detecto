import numpy as np


def pcd_to_numpy(pcd):
    """
    Convert Open3D point cloud to NumPy arrays.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    return points, colors
