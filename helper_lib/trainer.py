import json

import torch

from helper_lib.model import LATENT_DIM


def train_gan(model, data_loader, criterion, optimizers, device="cpu", epochs=10):
    """Train a standard GAN; optimizers are (generator, discriminator)."""
    optimizer_g, optimizer_d = optimizers
    model.to(device)
    history = []
    for epoch in range(epochs):
        model.train()
        total, total_g, total_d = 0, 0.0, 0.0
        for real, _ in data_loader:
            real = real.to(device)
            batch_size = real.size(0)
            true_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake = model.generator(noise)

            optimizer_d.zero_grad()
            loss_d = criterion(model.discriminator(real), true_labels)
            loss_d += criterion(model.discriminator(fake.detach()), fake_labels)
            if not torch.isfinite(loss_d).item():
                raise RuntimeError("Discriminator loss is not finite.")
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            loss_g = criterion(model.discriminator(fake), true_labels)
            if not torch.isfinite(loss_g).item():
                raise RuntimeError("Generator loss is not finite.")
            loss_g.backward()
            optimizer_g.step()

            total += batch_size
            total_d += loss_d.item() * batch_size
            total_g += loss_g.item() * batch_size

        row = {"epoch": epoch + 1, "discriminator_loss": total_d / total,
               "generator_loss": total_g / total}
        history.append(row)
        print(json.dumps(row), flush=True)
    return model, history
