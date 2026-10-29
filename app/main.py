from functools import lru_cache
from io import BytesIO
import pickle

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
import torch

from app.bigram_model import BigramModel
from app.cnn_model import CLASSES, image_transform, load_classifier
from helper_lib.generator import generate_samples
from helper_lib.model import load_generator

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another sentence in the corpus",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1)
    length: int = Field(ge=1, le=100)


@lru_cache(maxsize=1)
def load_embedding_model():
    import spacy

    return spacy.load("en_core_web_lg")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"generated_text": generated_text}


@app.get("/embedding")
def get_embedding(word: str = Query(min_length=1)):
    try:
        nlp = load_embedding_model()
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail="The en_core_web_lg spaCy model is not installed.",
        ) from exc

    doc = nlp(word.strip())
    if len(doc) != 1:
        raise HTTPException(status_code=422, detail="word must contain exactly one token")

    token = doc[0]
    if not token.has_vector:
        raise HTTPException(status_code=404, detail=f"No embedding found for '{word}'.")

    vector = token.vector.tolist()
    return {"word": token.text, "embedding": vector, "dimensions": len(vector)}


@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    content = file.file.read(5 * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The image file is empty.")
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The image must be at most 5 MiB.")
    try:
        with Image.open(BytesIO(content)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("The image has too many pixels.")
            inputs = image_transform(image.convert("RGB")).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Upload a valid image with at most 16 million pixels.") from exc

    try:
        model = load_classifier()
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise HTTPException(status_code=503, detail="The trained CNN weights are unavailable. Run train_cnn.py first.") from exc
    with torch.inference_mode():
        class_id = model(inputs).argmax(dim=1).item()
    return {"class_id": class_id, "class_name": CLASSES[class_id]}


@app.get("/generate-image", response_class=Response,
         responses={200: {"content": {"image/png": {}}}})
def generate_image(num_samples: int = Query(default=10, ge=1, le=16),
                   seed: int | None = Query(default=None, ge=0, le=2**63 - 1)):
    try:
        model = load_generator()
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise HTTPException(status_code=503, detail="The trained GAN weights are unavailable. Run train_gan.py first.") from exc
    image = generate_samples(model, num_samples=num_samples, seed=seed)
    output = BytesIO()
    image.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")
