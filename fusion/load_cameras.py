from pathlib import Path

def load_colmap_images(images_txt):
    cameras = {}

    with open(images_txt, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("#") or line == "":
            i += 1
            continue

        parts = line.split()

        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]

        cameras[image_name] = {
            "id": image_id,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "t": (tx, ty, tz),
            "camera_id": camera_id
        }

        i += 2

    return cameras


if __name__ == "__main__":
    cams = load_colmap_images(
        "reconstruction/colmap/scene2/images.txt"
    )

    print("Number of cameras:", len(cams))
    for k in list(cams.keys())[:3]:
        print(k, cams[k])
