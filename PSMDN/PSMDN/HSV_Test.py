import torch
import numpy as np
from PIL import Image


def hue(img, epsilon=1e-10):
    img = img.float()
    r, g, b = torch.split(img, 1, dim=1)

    max, arg_max = torch.max(img, dim=1, keepdim=True)
    min, arg_min = torch.min(img, dim=1, keepdim=True)

    max_min = max - min + epsilon

    h1 = 60.0 * (g - b) / max_min   # R
    h2 = 60.0 * (b - r) / max_min + 120.0  # G
    h3 = 60.0 * (r - g) / max_min + 240.0  # B

    h = torch.stack((h1, h2, h3), dim=1).gather(dim=1, index=arg_max.unsqueeze(0)).squeeze(1)
    return h, r, g, b


if __name__ == "__main__":
    loader = 'data/train_data/test/01.jpg'

    img = torch.tensor(np.asarray(Image.open(loader)))
    img = img.permute(2, 0, 1).unsqueeze(0)

    h, r, g, b = hue(img)
    print(h, r, g, b)

