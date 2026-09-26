FROM python:3.11-slim
WORKDIR /srv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
ENV PATH="/srv/.venv/bin:$PATH" HF_HOME=/srv/.hf
# bake the embedding model into the image so the container starts offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
COPY app ./app
# trained artifacts are mounted at /srv/artifacts (docker-compose) - produced by `python -m training.train_*`
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
