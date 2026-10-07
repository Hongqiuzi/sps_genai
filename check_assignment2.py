from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import torch

from app.cnn_model import CHECKPOINT, CLASSES, CNN, image_transform, load_classifier
from app.main import app


def main():
    torch.set_num_threads(4)
    model = CNN()
    assert sum(p.numel() for p in model.parameters()) == 825398
    assert (model.conv1.in_channels, model.conv1.out_channels) == (3, 16)
    assert (model.conv2.in_channels, model.conv2.out_channels) == (16, 32)
    for layer in (model.conv1, model.conv2):
        assert (layer.kernel_size, layer.stride, layer.padding) == ((3, 3), (1, 1), (1, 1))
    assert (model.pool.kernel_size, model.pool.stride) == (2, 2)
    assert (model.fc1.in_features, model.fc1.out_features, model.fc2.out_features) == (8192, 100, 10)
    x = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        assert model.pool(model.relu(model.conv1(x))).shape == (2, 16, 32, 32)
        assert model(x).shape == (2, 10)

    client = TestClient(app)
    assert client.get("/").json() == {"Hello": "World"}
    assert client.post("/generate", json={"start_word": "we", "length": 5}).json()["generated_text"].startswith("we ")
    embedding = client.get("/embedding", params={"word": "king"})
    assert embedding.status_code == 200
    assert embedding.json()["dimensions"] == len(embedding.json()["embedding"]) == 300
    assert CHECKPOINT.is_file(), "A real trained checkpoint is required."
    model = load_classifier()
    assert not model.training
    sample = CHECKPOINT.parents[1] / "assets" / "cifar10_sample.png"
    with Image.open(sample) as image, torch.inference_mode():
        inputs = image_transform(image.convert("RGB")).unsqueeze(0)
        expected = model(inputs).argmax(dim=1).item()
    response = client.post("/classify", files={"file": (sample.name, sample.read_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"class_id": expected, "class_name": CLASSES[expected]}
    assert client.post("/classify").status_code == 422
    assert client.post("/classify", files={"file": ("empty.png", b"", "image/png")}).status_code == 400
    assert client.post("/classify", files={"file": ("bad.png", b"not an image", "image/png")}).status_code == 400
    assert client.post("/classify", files={"file": ("large.png", b"x" * (5 * 1024 * 1024 + 1), "image/png")}).status_code == 413
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "weights.pt"
        torch.save(model.state_dict(), path)
        restored = CNN().eval()
        restored.load_state_dict(torch.load(path, weights_only=True))
        with torch.inference_mode():
            assert torch.equal(model(inputs), restored(inputs))
        load_classifier.cache_clear()
        with patch("app.cnn_model.CHECKPOINT", Path(temporary) / "missing.pt"):
            assert client.post("/classify", files={"file": ("sample.png", sample.read_bytes(), "image/png")}).status_code == 503
        load_classifier.cache_clear()
        path.write_bytes(b"invalid checkpoint")
        with patch("app.cnn_model.CHECKPOINT", path):
            assert client.post("/classify", files={"file": ("sample.png", sample.read_bytes(), "image/png")}).status_code == 503
        load_classifier.cache_clear()
    for mode in ("L", "RGBA"):
        stream = BytesIO()
        Image.new(mode, (32, 32)).save(stream, format="PNG")
        assert client.post("/classify", files={"file": ("sample.png", stream.getvalue(), "image/png")}).status_code == 200
    stream = BytesIO()
    Image.new("RGB", (32, 32)).save(stream, format="JPEG")
    assert client.post("/classify", files={"file": ("sample.jpg", stream.getvalue(), "image/jpeg")}).status_code == 200
    stream = BytesIO()
    Image.new("L", (4001, 4000)).save(stream, format="PNG")
    assert client.post("/classify", files={"file": ("pixels.png", stream.getvalue(), "image/png")}).status_code == 400
    print("Assignment 2 architecture, checkpoint, API, input-error and previous-endpoint checks passed.")


if __name__ == "__main__":
    main()
