from pathlib import Path
import open3d as o3d
from fusion.run_fusion import run_full_pipeline


def run_inspection(scene_dir: str, model_path: str):
    """
    Wrapper for Streamlit.
    """
    pcd = run_full_pipeline(
        scene_dir=scene_dir,
        model_path=model_path,
        clean_colmap=False
    )

    # Save colored point cloud
    output_dir = Path(scene_dir) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = output_dir / "crack_localization.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)

    return pcd, ply_path
