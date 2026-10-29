from math import ceil, sqrt

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from helper_lib.model import LATENT_DIM


def generate_samples(model, device="cpu", num_samples=10, seed=None):
    """Return a PNG-ready grid of images from the trained generator."""
    if not 1 <= num_samples <= 16:
        raise ValueError("num_samples must be between 1 and 16")
    generator = model.generator if hasattr(model, "generator") else model
    random_source = torch.Generator().manual_seed(seed) if seed is not None else None
    noise = torch.randn(num_samples, LATENT_DIM, generator=random_source).to(device)
    was_training = generator.training
    generator.eval()
    with torch.inference_mode():
        images = generator(noise).cpu()
    if was_training:
        generator.train()
    grid = make_grid((images + 1) / 2, nrow=ceil(sqrt(num_samples)), padding=2)
    return to_pil_image(grid)
