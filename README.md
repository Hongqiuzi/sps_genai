# APANPS 5560 Assignment 1

FastAPI project from the Module 3 class activity, extended with a spaCy word embedding endpoint for Assignment 1.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from this directory:

```powershell
uv sync
uv run fastapi dev app/main.py
```

Open <http://127.0.0.1:8000/docs> to try the API. The first install downloads spaCy's English large model, which is roughly 400 MB.

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
