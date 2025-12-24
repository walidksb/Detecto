import open3d as o3d
import numpy as np

from fusion.load_cameras import load_colmap_images


# --- paths ---
PCD_PATH = "reconstruction/output/scene2.ply"
IMAGES_TXT = "reconstruction/colmap/scene2/images.txt"


def main():
    # Load point cloud
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd.points)

    print("Point cloud loaded:", points.shape)

    # Load cameras
    cams = load_colmap_images(IMAGES_TXT)

    # Pick ONE image
    image_name = list(cams.keys())[0]
    cam = cams[image_name]

    cam_pos = np.array(cam["t"])
    print("Using image:", image_name)
    print("Camera position:", cam_pos)

    # Pick a few random points in space (for now)
    idx = np.random.choice(len(points), size=200, replace=False)
    selected = points[idx]

    # Color them red
    colors = np.zeros_like(points)
    colors[:] = [0.6, 0.6, 0.6]
    colors[idx] = [1.0, 0.0, 0.0]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
