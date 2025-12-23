import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from detection.dataset.crack_dataset import CrackDataset
from detection.training.model import SimpleUNet

def main():
    dataset = CrackDataset(
        images_dir="data/processed/crack500/images/train",
        masks_dir="data/processed/crack500/masks/train",
        image_size=512
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    model = SimpleUNet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    images, masks = next(iter(loader))

    preds = model(images)
    loss = criterion(preds, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("One training step completed ✅")
    print("Loss:", loss.item())

if __name__ == "__main__":
    main()
