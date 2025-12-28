import numpy as np
import open3d as o3d


PCD_PATH = "reconstruction/output/scene2.ply"
CONFIDENCE_PATH = "analysis/crack_confidence.npy"  # saved from Step 9.7
THRESHOLD = 0.6


def main():
    # Load point cloud
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd.points)

    # Load confidence values
    confidence = np.load(CONFIDENCE_PATH)

    assert len(points) == len(confidence)

    # Extract crack points
    crack_mask = confidence > THRESHOLD
    crack_points = points[crack_mask]

    print(f"Total points: {len(points)}")
    print(f"Crack points: {len(crack_points)}")

    # Create crack-only point cloud
    crack_pcd = o3d.geometry.PointCloud()
    crack_pcd.points = o3d.utility.Vector3dVector(crack_points)

    # Color cracks red
    crack_pcd.paint_uniform_color([1, 0, 0])

    # Visualize
    o3d.visualization.draw_geometries([crack_pcd])


if __name__ == "__main__":
    main()
