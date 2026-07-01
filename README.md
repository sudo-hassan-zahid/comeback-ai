# Comeback AI ⚡

**Your academic comeback, backed by data.**

Comeback AI identifies students who may need academic support, explains the strongest risk factors, and suggests practical next steps from a grounded knowledge base.

> This is a learning project trained on synthetic data. Do not use it for real academic, disciplinary, admissions, or medical decisions.

## Features

- Explainable student-risk prediction
- Logistic regression and random forest model comparison
- Local document retrieval with source attribution
- Optional Groq-powered answers
- Streamlit frontend
- FastAPI backend
- Docker support and automated tests

## Run everything with one command

Requirements: Python 3.11 or newer.

```powershell
python run.py
```

On its first run, this command automatically:

1. Creates `.venv`.
2. Installs all dependencies.
3. Trains the ML model.
4. Starts the API and frontend.
5. Opens the app in your browser.

Press `Ctrl+C` to stop everything.

Useful options:

```powershell
python run.py --retrain     # retrain the ML model
python run.py --setup       # reinstall dependencies
python run.py --no-browser  # do not open the browser
```

## Frontend

Yes—Comeback AI has a Streamlit frontend.

- App: <http://localhost:8501>
- API documentation: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

The frontend lets you enter a student profile, view the estimated support-risk and its main factors, and ask the grounded support guide questions.

## Optional Groq setup

The entire project works without Groq. To enable conversational answers:

1. Create an API key at <https://console.groq.com/keys>.
2. Copy `.env.example` to `.env`.
3. Add your key:

```dotenv
GROQ_API_KEY=your_new_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit `.env` or share an API key publicly.

## Testing—run in this order

From the project root in PowerShell:

### 1. Create the environment

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[ui,dev]"
```

### 2. Train and inspect the model

```powershell
.venv\Scripts\python -m comeback_ai.ml.train
Get-Content artifacts\model_card.json
```

Confirm that both model candidates have metrics and that `selected_model` is present.

### 3. Run automated tests

```powershell
.venv\Scripts\python -m pytest -q
```

Expected result: all tests pass.

### 4. Check code quality

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
```

Expected result: no lint or formatting errors.

### 5. Run the full application

```powershell
python run.py
```

Verify these flows in the frontend:

1. Submit a low-risk student profile.
2. Submit a high-risk student profile.
3. Confirm that each result includes explanatory factors.
4. Ask: `How can I catch up on missed assignments?`
5. Confirm that guidance includes local sources.

### 6. Test the API directly

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected response:

```text
status
------
healthy
```

## Docker alternative

```powershell
docker compose up --build
```

This starts the same frontend and API without requiring a local Python setup.

## Project structure

```text
src/comeback_ai/
├── api/        # FastAPI endpoints
├── domain/     # Request and response schemas
├── knowledge/  # Retrieval and optional Groq integration
└── ml/         # Data generation, training, and prediction
knowledge/      # Grounded support documents
tests/          # Automated tests
ui/             # Streamlit frontend
run.py          # One-command local launcher
```
