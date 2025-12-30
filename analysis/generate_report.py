import json
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


PCD_PATH = "reconstruction/output/scene2.ply"
CONFIDENCE_PATH = "analysis/crack_confidence.npy"

# Thresholds (inspection rules)
CONF_THRESHOLD = 0.6
EPS = 0.5
MIN_POINTS_CLUSTER = 10
MIN_LENGTH = 1.0
MIN_CRACKS_SCENE = 1

REPORT_PATH = "analysis/inspection_report.json"


def crack_length(points):
    if len(points) < 2:
        return 0.0
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return float(dists.max())


def main():
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd.points)
    confidence = np.load(CONFIDENCE_PATH)

    crack_mask = confidence > CONF_THRESHOLD
    crack_points = points[crack_mask]

    clustering = DBSCAN(eps=EPS, min_samples=MIN_POINTS_CLUSTER).fit(crack_points)
    labels = clustering.labels_

    clusters = {}
    for label in set(labels):
        if label == -1:
            continue
        clusters[label] = crack_points[labels == label]

    crack_reports = []

    for cid, pts in clusters.items():
        length = crack_length(pts)
        if length < MIN_LENGTH:
            continue

        center = pts.mean(axis=0).tolist()

        crack_reports.append({
            "id": int(cid),
            "length": round(length, 2),
            "num_points": int(len(pts)),
            "center_3d": [round(c, 3) for c in center]
        })

    scene_status = "DAMAGED" if len(crack_reports) >= MIN_CRACKS_SCENE else "NO_SIGNIFICANT_DAMAGE"

    report = {
        "scene_status": scene_status,
        "num_detected_cracks": len(crack_reports),
        "inspection_parameters": {
            "confidence_threshold": CONF_THRESHOLD,
            "min_crack_length": MIN_LENGTH,
            "min_cluster_points": MIN_POINTS_CLUSTER
        },
        "cracks": crack_reports
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("Inspection completed.")
    print(f"Scene status: {scene_status}")
    print(f"Detected cracks: {len(crack_reports)}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
