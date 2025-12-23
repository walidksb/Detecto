from pathlib import Path

def main():
    import sys
    print("Python:", sys.version)

    try:
        import torch
        print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    except Exception as e:
        print("Torch import failed:", e)

    for p in [
        Path("data/raw"), Path("data/processed"),
        Path("detection/models/checkpoints"), Path("detection/models/exported")
    ]:
        p.mkdir(parents=True, exist_ok=True)

    print("Smoke test OK ✅")

if __name__ == "__main__":
    main()
