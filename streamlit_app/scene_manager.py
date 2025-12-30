from pathlib import Path
from datetime import datetime


SCENES_ROOT = Path("runtime/scenes")


def create_new_scene():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_dir = SCENES_ROOT / f"scene_{timestamp}"
    images_dir = scene_dir / "images"

    images_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir, images_dir
