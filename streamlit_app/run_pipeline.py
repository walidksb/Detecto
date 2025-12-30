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
    return pcd
