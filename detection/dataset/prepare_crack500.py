from pathlib import Path
import shutil
import random

RAW = Path("data/raw/crack500")
OUT = Path("data/processed/crack500")

def main():
    images = sorted((RAW / "images").glob("*"))
    masks = sorted((RAW / "masks").glob("*"))

    assert len(images) == len(masks), "Images / masks mismatch"

    pairs = list(zip(images, masks))
    random.shuffle(pairs)

    split = int(0.8 * len(pairs))
    train, val = pairs[:split], pairs[split:]

    for split_name, split_data in [("train", train), ("val", val)]:
        for img, msk in split_data:
            (OUT / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (OUT / "masks" / split_name).mkdir(parents=True, exist_ok=True)

            shutil.copy(img, OUT / "images" / split_name / img.name)
            shutil.copy(msk, OUT / "masks" / split_name / msk.name)

    print("Crack500 prepared ✅")

if __name__ == "__main__":
    main()
