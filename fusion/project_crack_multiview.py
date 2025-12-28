import cv2
import numpy as np
import open3d as o3d

from fusion.load_cameras import load_colmap_images
from fusion.load_intrinsics import load_colmap_intrinsics
from detection.inference.predict_mask import load_model, predict_crack_mask


PCD_PATH = "reconstruction/output/scene2.ply"
IMAGES_TXT = "reconstruction/colmap/scene2/images.txt"
CAMERAS_TXT = "reconstruction/colmap/scene2/cameras.txt"
IMAGE_DIR = "reconstruction/images/scene2"
MODEL_PATH = "detection/models/exported/crack_unet_v1.pth"


def main():
    # Load point cloud
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd.points)
    n_points = len(points)

    # Load cameras & intrinsics
    cams = load_colmap_images(IMAGES_TXT)
    intr = load_colmap_intrinsics(CAMERAS_TXT)

    # Load model
    model = load_model(MODEL_PATH)

    # Initialize fusion counter
    votes = np.zeros(n_points)

    for image_name, cam in cams.items():
        img = cv2.imread(f"{IMAGE_DIR}/{image_name}")
        if img is None:
            continue

        h, w = img.shape[:2]

        mask = predict_crack_mask(model, img)

        cam_intr = intr[cam["camera_id"]]
        fx, fy = cam_intr["fx"], cam_intr["fy"]
        cx, cy = cam_intr["cx"], cam_intr["cy"]

        for i, (X, Y, Z) in enumerate(points):
            if Z <= 0:
                continue

            u = int(fx * X / Z + cx)
            v = int(fy * Y / Z + cy)

            if 0 <= u < w and 0 <= v < h:
                if mask[v, u] > 0:
                    votes[i] += 1

    # Normalize votes
    votes /= votes.max() + 1e-6
    
    np.save("analysis/crack_confidence.npy", votes)

    # Colorize
    colors = np.zeros((n_points, 3))
    for i, v in enumerate(votes):
        colors[i] = [v, 0, 1 - v]  # blue → red

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
