import torch
import cv2
from detection.training.model import SimpleUNet

model = SimpleUNet()
model.load_state_dict(
    torch.load("detection/models/exported/crack_unet_v1.pth", map_location="cpu")
)
model.eval()

img = cv2.imread("data/processed/crack500/images/val/" +
                  sorted(__import__("os").listdir("data/processed/crack500/images/val"))[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))
img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
img = img.unsqueeze(0)

with torch.no_grad():
    pred = model(img)

print("Inference output shape:", pred.shape)
print("Min / Max:", pred.min().item(), pred.max().item())
