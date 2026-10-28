from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import torch
from torch import nn

from app.main import app
from app.cnn_model import CLASSES, CHECKPOINT as CNN_CHECKPOINT, image_transform, load_classifier
from helper_lib.generator import generate_samples
from helper_lib.model import CHECKPOINT, Discriminator, Generator, get_model, load_generator


def main():
    torch.set_num_threads(4)
    gan = get_model("GAN")
    assert isinstance(gan.generator, Generator)
    assert isinstance(gan.discriminator, Discriminator)
    assert (gan.generator.fc.in_features, gan.generator.fc.out_features) == (100, 6272)
    assert (gan.discriminator.fc.in_features, gan.discriminator.fc.out_features) == (6272, 1)
    for layer in (gan.generator.deconv1, gan.generator.deconv2):
        assert (layer.kernel_size, layer.stride, layer.padding) == ((4, 4), (2, 2), (1, 1))
    for layer in (gan.discriminator.conv1, gan.discriminator.conv2):
        assert (layer.kernel_size, layer.stride, layer.padding) == ((4, 4), (2, 2), (1, 1))
    noise = torch.randn(2, 100)
    fake = gan.generator(noise)
    assert fake.shape == (2, 1, 28, 28)
    assert gan.discriminator(fake).shape == (2, 1)
    assert fake.min() >= -1 and fake.max() <= 1
    assert all(0 <= p <= 1 for p in gan.discriminator(fake).flatten().tolist())
    nn.BCELoss()(gan.discriminator(fake), torch.ones(2, 1)).backward()
    assert gan.generator.fc.weight.grad is not None
    assert torch.isfinite(gan.generator.fc.weight.grad).all()

    assert CHECKPOINT.is_file(), "A real MNIST-trained generator checkpoint is required."
    trained = load_generator()
    assert not trained.training
    memory = BytesIO()
    torch.save(trained.state_dict(), memory)
    memory.seek(0)
    restored = Generator().eval()
    restored.load_state_dict(torch.load(memory, weights_only=True))
    with torch.inference_mode():
        assert torch.equal(trained(noise), restored(noise))

    client = TestClient(app)
    response = client.get("/generate-image", params={"num_samples": 4, "seed": 42})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == (62, 62)
        image.verify()
    assert response.content == client.get("/generate-image", params={"num_samples": 4, "seed": 42}).content
    assert client.get("/generate-image", params={"num_samples": 0}).status_code == 422
    assert client.get("/generate-image", params={"num_samples": 17}).status_code == 422
    assert client.get("/generate-image", params={"seed": -1}).status_code == 422
    assert client.get("/").json() == {"Hello": "World"}
    assert client.post("/generate", json={"start_word": "we", "length": 5}).status_code == 200
    embedding = client.get("/embedding", params={"word": "king"})
    assert embedding.status_code == 200
    assert embedding.json()["dimensions"] == 300
    sample = CNN_CHECKPOINT.parents[1] / "assets" / "cifar10_sample.png"
    with Image.open(sample) as image, torch.inference_mode():
        expected = load_classifier()(image_transform(image.convert("RGB")).unsqueeze(0)).argmax(dim=1).item()
    response = client.post("/classify", files={"file": (sample.name, sample.read_bytes(), "image/png")})
    assert response.json() == {"class_id": expected, "class_name": CLASSES[expected]}

    load_generator.cache_clear()
    with patch("helper_lib.model.CHECKPOINT", CHECKPOINT.parent / "missing.pt"):
        assert client.get("/generate-image").status_code == 503
    load_generator.cache_clear()

    grid = generate_samples(trained, num_samples=16, seed=42)
    assert grid.size == (122, 122)
    print("Assignment 3 architecture, checkpoint, PNG API and previous-route checks passed.")


if __name__ == "__main__":
    main()
