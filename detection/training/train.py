import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from detection.dataset.crack_dataset import CrackDataset
from detection.training.model import SimpleUNet


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/train_detection.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = CrackDataset(
        cfg["data"]["train_images"],
        cfg["data"]["train_masks"],
        cfg["training"]["image_size"]
    )

    val_ds = CrackDataset(
        cfg["data"]["val_images"],
        cfg["data"]["val_masks"],
        cfg["training"]["image_size"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )

    model = SimpleUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"]
    )

    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_loss += criterion(preds, masks).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        torch.save(
            model.state_dict(),
            ckpt_dir / f"model_epoch_{epoch+1}.pth"
        )

    print("Training completed ✅")


if __name__ == "__main__":
    main()
