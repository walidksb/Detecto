import subprocess
from pathlib import Path
import shutil


def run_cmd(cmd: list):
    """Run a shell command and raise error if it fails."""
    print("▶ Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError("Command failed")


def run_colmap(scene_dir: str, image_dir_name="images", clean=False):
    """
    Run COLMAP sparse reconstruction inside Docker.

    Returns a dict with paths to reconstruction outputs.
    """
    scene_dir = Path(scene_dir).resolve()
    images_dir = scene_dir / image_dir_name
    colmap_dir = scene_dir / "colmap"
    sparse_root = colmap_dir / "sparse"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Optional cleanup
    if clean and colmap_dir.exists():
        print("🧹 Cleaning previous COLMAP results")
        shutil.rmtree(colmap_dir)

    colmap_dir.mkdir(exist_ok=True)
    sparse_root.mkdir(exist_ok=True)

    volume = f"{scene_dir}:/scene"

    # 1. Feature extraction
    run_cmd([
        "docker", "run", "--rm",
        "-v", volume,
        "colmap:cpu",
        "colmap", "feature_extractor",
        "--database_path", "/scene/colmap/database.db",
        "--image_path", "/scene/images",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.num_threads", "2",
    ])

    # 2. Feature matching
    run_cmd([
        "docker", "run", "--rm",
        "-v", volume,
        "colmap:cpu",
        "colmap", "exhaustive_matcher",
        "--database_path", "/scene/colmap/database.db",
        "--SiftMatching.use_gpu", "0",
    ])

    # 3. Sparse mapping
    run_cmd([
        "docker", "run", "--rm",
        "-v", volume,
        "colmap:cpu",
        "colmap", "mapper",
        "--database_path", "/scene/colmap/database.db",
        "--image_path", "/scene/images",
        "--output_path", "/scene/colmap/sparse",
    ])

    # 4. Convert sparse model to TXT (for loaders)
    sparse_txt = colmap_dir / "sparse_txt"
    sparse_txt.mkdir(exist_ok=True)

    run_cmd([
        "docker", "run", "--rm",
        "-v", volume,
        "colmap:cpu",
        "colmap", "model_converter",
        "--input_path", "/scene/colmap/sparse/0",
        "--output_path", "/scene/colmap/sparse_txt",
        "--output_type", "TXT",
    ])

    # 5. Export sparse point cloud as PLY
    run_cmd([
        "docker", "run", "--rm",
        "-v", volume,
        "colmap:cpu",
        "colmap", "model_converter",
        "--input_path", "/scene/colmap/sparse/0",
        "--output_path", "/scene/colmap/sparse.ply",
        "--output_type", "PLY",
    ])


    # Locate sparse model
    sparse_txt = colmap_dir / "sparse_txt"

    result = {
        "scene_dir": scene_dir,
        "colmap_dir": colmap_dir,
        "database": colmap_dir / "database.db",
        "sparse_dir": sparse_txt,
        "cameras": sparse_txt / "cameras.txt",
        "images": sparse_txt / "images.txt",
        "points3D": sparse_txt / "points3D.txt",
        "ply": colmap_dir / "sparse.ply",
    }

    print("✅ COLMAP sparse reconstruction completed")
    return result
