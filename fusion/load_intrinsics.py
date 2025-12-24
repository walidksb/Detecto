def load_colmap_intrinsics(cameras_txt):
    intrinsics = {}

    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue

            parts = line.split()

            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])

            if model == "PINHOLE":
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])

            elif model == "SIMPLE_PINHOLE":
                fx = fy = float(parts[4])
                cx = float(parts[5])
                cy = float(parts[6])

            elif model == "SIMPLE_RADIAL":
                fx = fy = float(parts[4])
                cx = float(parts[5])
                cy = float(parts[6])

            else:
                raise ValueError(f"Unsupported camera model: {model}")

            intrinsics[cam_id] = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
                "model": model
            }

    return intrinsics


if __name__ == "__main__":
    intr = load_colmap_intrinsics(
        "reconstruction/colmap/scene2/cameras.txt"
    )

    for k, v in intr.items():
        print("Camera", k, v)
