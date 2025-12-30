import numpy as np
import pandas as pd

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

def export_confidence_csv(npy_path, csv_path):
    votes = np.load(npy_path)
    df = pd.DataFrame({
        "point_id": range(len(votes)),
        "crack_confidence": votes
    })
    df.to_csv(csv_path, index=False)