import torch
import cv2
import numpy as np

from detection.training.model import SimpleUNet


def load_model(weights_path):
    model = SimpleUNet()
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu")
    )
    model.eval()
    return model


def predict_crack_mask(model, image_bgr, image_size=512, threshold=0.5):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0, 0].numpy()

    pred = (pred > threshold).astype(np.uint8) * 255
    pred = cv2.resize(pred, (image_bgr.shape[1], image_bgr.shape[0]))

    return pred
