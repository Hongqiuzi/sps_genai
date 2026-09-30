from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
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
