import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


PCD_PATH = "reconstruction/output/scene2.ply"
CONFIDENCE_PATH = "analysis/crack_confidence.npy"

# Thresholds
CONF_THRESHOLD = 0.6
EPS = 0.5
MIN_POINTS = 5
MIN_LENGTH = 0.5   # minimum crack length (units of point cloud)


def crack_length(points):
    # Approximate crack length as max pairwise distance
    if len(points) < 2:
        return 0.0

    diffs = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return dists.max()


def main():
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd.points)
    confidence = np.load(CONFIDENCE_PATH)

    crack_mask = confidence > CONF_THRESHOLD
    crack_points = points[crack_mask]

    clustering = DBSCAN(eps=EPS, min_samples=MIN_POINTS).fit(crack_points)
    labels = clustering.labels_

    clusters = {}
    for label in set(labels):
        if label == -1:
            continue
        clusters[label] = crack_points[labels == label]

    print(f"Total clusters: {len(clusters)}")

    significant = []

    for cid, pts in clusters.items():
        length = crack_length(pts)
        if length >= MIN_LENGTH:
            significant.append((cid, length, len(pts)))

    print(f"Significant cracks: {len(significant)}\n")

    for cid, length, npts in significant:
        print(f"Crack {cid}: length={length:.2f}, points={npts}")


if __name__ == "__main__":
    main()
