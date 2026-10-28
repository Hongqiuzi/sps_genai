# APANPS 5560 - Assignments 1, 2 and 3

I extended the class FastAPI project with word embeddings for Assignment 1, a CIFAR-10 image classifier for Assignment 2, and an MNIST digit generator for Assignment 3.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from this directory:

```powershell
uv sync
uv run fastapi dev app/main.py
```

Open <http://127.0.0.1:8000/docs> to try the API. The first install downloads spaCy's English large model and CPU versions of PyTorch and torchvision. The trained CNN weights are included, so retraining is not required to start the API.

## Run with Docker

```powershell
docker build -t sps-genai .
docker run --rm -p 8000:80 sps-genai
```

Open <http://127.0.0.1:8000/docs> after the container starts.

## API calls

```powershell
curl.exe http://127.0.0.1:8000/
curl.exe -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{"start_word":"we","length":5}'
curl.exe "http://127.0.0.1:8000/embedding?word=king"
```

`POST /generate` returns text from a simple bigram model. `GET /embedding` returns a word's 300-dimensional vector from `en_core_web_lg`, with the word and vector length.

## Assignment 2 CNN

I used the architecture specified in the assignment:

| Operation | Output shape, excluding the batch dimension |
| --- | --- |
| RGB input | 3 x 64 x 64 |
| Conv2d(3, 16, kernel=3, stride=1, padding=1), ReLU | 16 x 64 x 64 |
| MaxPool2d(kernel=2, stride=2) | 16 x 32 x 32 |
| Conv2d(16, 32, kernel=3, stride=1, padding=1), ReLU | 32 x 32 x 32 |
| MaxPool2d(kernel=2, stride=2) | 32 x 16 x 16 |
| Flatten | 8192 |
| Linear(8192, 100), ReLU | 100 |
| Linear(100, 10) | 10 |

CIFAR-10 images are originally 32 x 32. I resize them to 64 x 64 to match the assignment input. Training and prediction share the same transform: resize, convert to a tensor, and normalize each RGB channel with mean 0.5 and standard deviation 0.5. The model returns raw logits; CrossEntropyLoss receives these directly, and argmax selects the predicted class.

### Reproduce training

```powershell
uv run python train_cnn.py
```

The default run uses all 50,000 training images for five epochs, Adam with learning rate 0.001, batch size 64, seed 42, and CPU. The 10,000 test images are evaluated after training and are not used to select settings. It saves `models/cifar10_cnn.pt`, `training_results.json`, and the first test image as `assets/cifar10_sample.png`. The results file contains measured losses, accuracies and the sample prediction. No pretrained model or data augmentation is used.

For the included checkpoint, test accuracy was **65.45%** and test loss was **1.0203**. The first test image has the label `cat`, and the model predicted `cat`.

### Classify an image

```powershell
curl.exe -X POST "http://127.0.0.1:8000/classify" -F "file=@assets/cifar10_sample.png"
```

The multipart field is named `file`. The response contains `class_id` and `class_name`. Class IDs 0-9 correspond to airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Uploaded images are converted to RGB before preprocessing.

The API returns 422 for a missing file, 400 for an empty or invalid image, 413 for a file above 5 MiB, and 503 if the trained weights are unavailable. Images must have at most 16 million pixels. Inference uses `model.eval()` and `torch.inference_mode()`.

### Check the implementation

```powershell
uv run python check_assignment2.py
```

This checks the architecture, batch shape, checkpoint reload, real image inference, input errors and previous endpoints. The Docker workflow calls the running container over HTTP and compares classification with the recorded sample prediction. Accuracy is assessed separately using the full test set; a working API does not imply every prediction is correct.

## Assignment 3 GAN

I implemented the architecture in Assignment 3. The generator maps a 100-value noise vector to a 28 x 28 grayscale image. The discriminator returns one real-image probability. I trained a standard GAN with binary cross-entropy: the discriminator learns real=1 and fake=0, while the generator tries to make its images receive real=1. The Module 6 practical used a Wasserstein GAN on CelebA; this assignment specifies a probability output and MNIST, so I used the assignment architecture and loss.

MNIST is loaded with `torchvision.datasets.MNIST(train=True, download=True)`. I use all 60,000 training images, which are already 28 x 28 grayscale, and do not resize them. `ToTensor()` converts pixels to 0-1; normalization with mean 0.5 and standard deviation 0.5 maps them to -1 to 1 for the generator's Tanh output. The digit labels are not used by this unconditional GAN, so its API cannot request a particular digit.

### Reproduce training

```powershell
uv run python train_gan.py
```

The defaults are seed 42, batch size 128, 10 epochs, CPU, and Adam for each network with learning rate 0.0002 and betas (0.5, 0.999). Training saves `models/mnist_gan_generator.pt`, `gan_training_results.json`, and a 16-image sample grid at `assets/mnist_gan_sample.png`. The included checkpoint is used by the API without retraining. The training results file records the actual losses; GAN losses are not classification accuracy.

### Generate images through the API

```powershell
curl.exe "http://127.0.0.1:8000/generate-image?num_samples=10&seed=42" --output digits.png
```

`GET /generate-image` returns a PNG grid of 1-16 generated images. `seed` is optional; using the same seed and checkpoint produces the same image. The API returns 422 for an invalid count or seed, and 503 if the trained generator weights are unavailable. To check the implementation after training, run `uv run python check_assignment3.py` and `uv run python check_assignment2.py`.
