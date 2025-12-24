import cv2
import numpy as np
import open3d as o3d

from fusion.load_cameras import load_colmap_images
from detection.inference.predict_mask import load_model, predict_crack_mask
from fusion.load_intrinsics import load_colmap_intrinsics


PCD_PATH = "reconstruction/output/scene2.ply"
IMAGES_TXT = "reconstruction/colmap/scene2/images.txt"
IMAGE_DIR = "reconstruction/images/scene2"
IMAGE_NAME = "DSC_0317.JPG"
MODEL_PATH = "detection/models/exported/crack_unet_v1.pth"




def main():
    # Load point cloud
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd.points)

    # Load cameras
    cams = load_colmap_images(IMAGES_TXT)
    cam = cams[IMAGE_NAME]

    # Load image
    img = cv2.imread(f"{IMAGE_DIR}/{IMAGE_NAME}")
    h, w = img.shape[:2]

    # Load model and predict real crack mask
    model = load_model(MODEL_PATH)
    mask = predict_crack_mask(model, img)

    intrinsics = load_colmap_intrinsics("reconstruction/colmap/scene2/cameras.txt")

    cam_intr = intrinsics[cam["camera_id"]]

    fx = cam_intr["fx"]
    fy = cam_intr["fy"]
    cx = cam_intr["cx"]
    cy = cam_intr["cy"]

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
