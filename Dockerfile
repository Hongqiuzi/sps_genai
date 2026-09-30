FROM ghcr.io/astral-sh/uv:0.12.21 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
ENV UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=0

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
EXPOSE 80
CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "80"]
