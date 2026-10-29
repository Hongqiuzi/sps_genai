import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from app.cnn_model import CHECKPOINT, CLASSES, CNN, image_transform


def train_model(model, train_loader, criterion, optimizer, epochs):
    history = []
    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training produced a non-finite loss.")
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        row = {
            "epoch": epoch + 1, "train_loss": running_loss / total,
            "train_accuracy": correct / total,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return {"test_loss": running_loss / total, "test_accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser(description="Train the Assignment 2 CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    torch.manual_seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    root = Path(__file__).resolve().parent
    train_data = CIFAR10(root / "data", train=True, download=True, transform=image_transform)
    test_data = CIFAR10(root / "data", train=False, download=True, transform=image_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = train_model(model, train_loader, criterion, optimizer, args.epochs)
    result = evaluate_model(model, test_loader, criterion)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(CHECKPOINT)

    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    sample = Image.fromarray(test_data.data[0])
    sample.save(assets / "cifar10_sample.png")
    with torch.inference_mode():
        sample_prediction = model(image_transform(sample).unsqueeze(0)).argmax(dim=1).item()
    result.update({
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": 0.001,
        "optimizer": "Adam", "seed": 42, "device": "cpu",
        "train_examples": len(train_data), "test_examples": len(test_data),
        "parameters": sum(p.numel() for p in model.parameters()), "history": history,
        "sample": {"test_index": 0, "true_class_id": int(test_data.targets[0]),
                   "true_class_name": CLASSES[test_data.targets[0]],
                   "predicted_class_id": sample_prediction,
                   "predicted_class_name": CLASSES[sample_prediction]},
    })
    (root / "training_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
