.PHONY: run install train api ui test lint docker

run:
	python run.py

install:
	python -m pip install -e ".[ui,dev]"
train:
	python -m comeback_ai.ml.train
api:
	uvicorn comeback_ai.api.main:app --reload
ui:
	streamlit run ui/app.py
test:
	pytest -q
lint:
	ruff check .
docker:
	docker compose up --build
