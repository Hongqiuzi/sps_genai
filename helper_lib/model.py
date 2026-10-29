from functools import lru_cache
from pathlib import Path

import torch
from torch import nn


LATENT_DIM = 100
CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "mnist_gan_generator.pt"


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 128 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        x = self.fc(noise).view(-1, 128, 7, 7)
        x = self.relu(self.bn1(self.deconv1(x)))
        return self.tanh(self.deconv2(x))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.leaky_relu(self.conv1(image))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        return self.sigmoid(self.fc(self.flatten(x)))


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()


def get_model(model_name):
    if model_name != "GAN":
        raise ValueError("model_name must be 'GAN'")
    return GAN()


@lru_cache(maxsize=1)
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.eval()
    return model
