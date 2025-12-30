import numpy as np


def pcd_to_numpy(pcd):
    """
    Convert Open3D point cloud to NumPy arrays.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    return points, colors

def crack_stats(confidence_path, threshold=0.3):
    votes = np.load(confidence_path)

    crack_points = votes > threshold
    total_points = len(votes)

    return {
        "total_points": total_points,
        "crack_points": int(crack_points.sum()),
        "crack_ratio": float(crack_points.sum() / total_points),
        "max_confidence": float(votes.max()),
        "mean_confidence": float(votes.mean()),
    }