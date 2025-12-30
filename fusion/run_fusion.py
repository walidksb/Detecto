from pathlib import Path
import open3d as o3d

from reconstruction.colmap_runner import run_colmap
from fusion.project_crack_multiview import project_cracks_multiview


def run_full_pipeline(scene_dir: str, model_path: str, clean_colmap=False):
    """
    Full pipeline:
    images → COLMAP → sparse PLY → multi-view crack fusion
    """
    scene_dir = Path(scene_dir).resolve()

    # 1. Run COLMAP (sparse + TXT + PLY)
    colmap_res = run_colmap(scene_dir, clean=clean_colmap)

    # 2. Run crack projection & fusion in 3D
    pcd = project_cracks_multiview(
        pcd_path=colmap_res["ply"],
        images_txt=colmap_res["images"],
        cameras_txt=colmap_res["cameras"],
        image_dir=scene_dir / "images",
        model_path=model_path,
    )

    return pcd
