import cv2
import numpy as np
import open3d as o3d

from fusion.load_cameras import load_colmap_images


PCD_PATH = "reconstruction/output/scene2.ply"
IMAGES_TXT = "reconstruction/colmap/scene2/images.txt"
IMAGE_PATH = "reconstruction/images/scene2"
IMAGE_NAME = "DSC_0317.JPG"


def main():
    # Load point cloud
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd.points)

    # Load cameras
    cams = load_colmap_images(IMAGES_TXT)
    cam = cams[IMAGE_NAME]

    # Load image
    img = cv2.imread(f"{IMAGE_PATH}/{IMAGE_NAME}")
    h, w = img.shape[:2]

    # Fake crack mask (for now)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Simple pinhole projection (VERY simplified)
    fx = fy = 1000
    cx, cy = w / 2, h / 2

    colors = np.zeros_like(points)
    colors[:] = [0.6, 0.6, 0.6]

    for i, p in enumerate(points):
        X, Y, Z = p
        if Z <= 0:
            continue

        u = int(fx * X / Z + cx)
        v = int(fy * Y / Z + cy)

        if 0 <= u < w and 0 <= v < h:
            if mask[v, u] > 0:
                colors[i] = [1, 0, 0]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
