FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY knowledge ./knowledge
COPY ui ./ui
RUN pip install --no-cache-dir ".[ui]"
RUN python -m comeback_ai.ml.train

EXPOSE 8000
CMD ["uvicorn", "comeback_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
