# Comeback AI ⚡

**Your academic comeback, backed by data.** Comeback AI is a local-first student-success project that combines foundational machine learning with grounded AI guidance. It estimates when a student may need extra academic support, explains the strongest factors behind the signal, and retrieves practical actions from a curated knowledge base.

It is designed as a serious learning project: the model is trained from scratch, two algorithms are compared, evaluation metrics are saved in a model card, retrieval works locally, and an LLM is an optional enhancement rather than a dependency.

> **Important:** the included dataset is synthetic. This is a portfolio and learning system, not a tool for grading, admissions, discipline, diagnosis, or real student decisions.

## What you will learn

- **ML foundations:** feature design, train/test splitting, preprocessing, logistic regression, random forests, class balance, precision, recall, F1, and ROC-AUC.
- **Explainability:** turning model behavior into understandable risk-increasing and risk-reducing factors.
- **RAG foundations:** document chunking, TF-IDF vectors, similarity search, grounded context, and source display.
- **AI engineering:** FastAPI, typed schemas, service boundaries, tests, environment configuration, Docker, and a Streamlit UI.

## Architecture

```text
Streamlit UI ───────► FastAPI
                       ├──► ML risk service ──► trained sklearn pipeline
                       └──► guidance service ─► local TF-IDF retrieval
                                                └──► optional Groq LLM
```

The default path is completely local and requires no account, API key, vector database, or paid subscription.

## Quick start with Docker

You only need [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
docker compose up --build
```

Then open:

- App: <http://localhost:8501>
- Interactive API docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

The image trains and evaluates the ML candidates during the build. Metrics and limitations are written to `artifacts/model_card.json` inside the API container.

## Local Python setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[ui,dev]"
python -m comeback_ai.ml.train
```

Use separate terminals for:

```bash
uvicorn comeback_ai.api.main:app --reload
streamlit run ui/app.py
```

Run quality checks with:

```bash
ruff check .
pytest -q
```

## Optional free Groq setup

Without Groq, Comeback AI returns the most relevant passages directly. Groq only rewrites those retrieved passages into a more conversational answer.

1. Visit <https://console.groq.com/> and create an account.
2. Open **API Keys**, create a key, and copy `.env.example` to `.env`.
3. Add the key locally:

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit `.env`; it is already ignored. Free-tier limits and available models can change, so check Groq's console before relying on it. If the key is absent, all features still work.

## Project structure

```text
src/comeback_ai/
├── api/           # HTTP endpoints and application lifecycle
├── domain/        # Validated request/response contracts
├── knowledge/     # Local retrieval and optional LLM service
└── ml/            # Synthetic data, training, evaluation, inference
knowledge/         # Human-editable grounded support documents
ui/                # Streamlit web application
tests/             # Data, retrieval, model, and API tests
```

## ML workflow

`python -m comeback_ai.ml.train` generates reproducible synthetic examples, creates a stratified split, and compares class-balanced logistic regression with a random forest. The candidate with the best held-out F1 score is saved with its full preprocessing pipeline. The generated model card records every candidate's precision, recall, F1, and ROC-AUC.

Synthetic data is intentional for a zero-cost, privacy-safe starter. The meaningful next phase is to replace it with a consented or public education dataset, document its license, audit performance across relevant groups, choose thresholds with educators, and add drift monitoring.

## API example

```bash
curl -X POST http://localhost:8000/v1/risk \
  -H "Content-Type: application/json" \
  -d '{
    "attendance_rate": 72,
    "assignment_completion": 65,
    "average_grade": 58,
    "study_hours_weekly": 7,
    "previous_failures": 1,
    "commute_minutes": 45,
    "has_internet": true,
    "works_part_time": false,
    "reports_stress": true,
    "asked_for_help": false
  }'
```

## Responsible roadmap

1. Replace synthetic data with a properly licensed dataset and add a data card.
2. Add fairness slices and threshold analysis—not sensitive attributes as automatic decision inputs.
3. Let institutions author and approve their own support knowledge.
4. Add feedback and retrieval evaluation before switching to a hosted vector store.
5. Only add Pinecone or another service when corpus size proves local retrieval insufficient.

That order keeps the project educational, useful, and genuinely maintainable instead of collecting services for their own sake.
