import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

# Copied from ComfyUI-MVAdapter/utils.py
def convert_images_to_tensors(images: list[Image.Image]):
    """Converts a list of PIL Images to a ComfyUI-style tensor (BHWC)."""
    return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])

# Copied from ComfyUI-MVAdapter/utils.py
def convert_tensors_to_images(images: torch.tensor):
    """Converts a ComfyUI-style tensor (BHWC) to a list of PIL Images."""
    return [
        Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
        for image in images
    ]
