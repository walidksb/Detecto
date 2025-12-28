import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


PCD_PATH = "reconstruction/output/scene2.ply"
CONFIDENCE_PATH = "analysis/crack_confidence.npy"
THRESHOLD = 0.6

# Clustering parameters
EPS = 0.2      # distance threshold (adjust if needed)
MIN_POINTS = 5 # minimum points to form a crack


def main():
    # Load point cloud
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd.points)

    # Load confidence
    confidence = np.load(CONFIDENCE_PATH)

    # Extract crack points
    crack_mask = confidence > THRESHOLD
    crack_points = points[crack_mask]

    print("Crack points min:", crack_points.min(axis=0))
    print("Crack points max:", crack_points.max(axis=0))
    print(f"Crack points: {len(crack_points)}")

    # Run DBSCAN clustering
    clustering = DBSCAN(eps=EPS, min_samples=MIN_POINTS).fit(crack_points)
    labels = clustering.labels_

    # Number of clusters (-1 is noise)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

    print(f"Detected cracks: {n_clusters}")

    # Color clusters
    colors = np.zeros((len(crack_points), 3))
    for label in unique_labels:
        if label == -1:
            colors[labels == label] = [0.5, 0.5, 0.5]
            continue

        color = np.random.rand(3)
        colors[labels == label] = color

    # Create colored crack cloud
    crack_pcd = o3d.geometry.PointCloud()
    crack_pcd.points = o3d.utility.Vector3dVector(crack_points)
    crack_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([crack_pcd])


if __name__ == "__main__":
    main()
