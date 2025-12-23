from detection.dataset.crack_dataset import CrackDataset

ds = CrackDataset(
    images_dir="data/processed/crack500/images/train",
    masks_dir="data/processed/crack500/masks/train"
)

print("Dataset size:", len(ds))
x, y = ds[0]
print("Image shape:", x.shape)
print("Mask shape:", y.shape)
print("Image min/max:", x.min().item(), x.max().item())
print("Mask min/max:", y.min().item(), y.max().item())
