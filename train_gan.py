import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.generator import generate_samples
from helper_lib.model import CHECKPOINT, LATENT_DIM, get_model
from helper_lib.trainer import train_gan


def main():
    parser = argparse.ArgumentParser(description="Train the Assignment 3 GAN on MNIST.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    torch.manual_seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    root = Path(__file__).resolve().parent
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_data = datasets.MNIST(root / "data", train=True, download=True, transform=transform)
    assert len(train_data) == 60000
    first_image, _ = train_data[0]
    assert first_image.shape == (1, 28, 28)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    model = get_model("GAN")
    criterion = nn.BCELoss()
    optimizers = (
        optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
    )
    model, history = train_gan(model, train_loader, criterion, optimizers, epochs=args.epochs)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT.with_suffix(".tmp")
    torch.save(model.generator.state_dict(), temporary)
    temporary.replace(CHECKPOINT)
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    sample_path = assets / "mnist_gan_sample.png"
    generate_samples(model, num_samples=16, seed=42).save(sample_path)
    result = {
        "dataset": "torchvision.datasets.MNIST (training split)",
        "train_examples": len(train_data),
        "image_shape": [1, 28, 28],
        "normalization": "(pixel / 255 - 0.5) / 0.5",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": 42,
        "device": "cpu",
        "optimizer": "Adam",
        "learning_rate": 0.0002,
        "betas": [0.5, 0.999],
        "latent_dim": LATENT_DIM,
        "generator_parameters": sum(p.numel() for p in model.generator.parameters()),
        "discriminator_parameters": sum(p.numel() for p in model.discriminator.parameters()),
        "history": history,
        "sample_file": "assets/mnist_gan_sample.png",
        "sample_seed": 42,
    }
    (root / "gan_training_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
