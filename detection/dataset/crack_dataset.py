from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset

class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=512):
        self.images = sorted(Path(images_dir).glob("*"))
        self.masks = sorted(Path(masks_dir).glob("*"))
        self.image_size = image_size

        assert len(self.images) == len(self.masks), \
            "Images and masks count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return image, mask
