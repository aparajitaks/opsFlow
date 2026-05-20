# ==========================================
# Makefile - opsFlow Automation Command Hooks
# ==========================================

.PHONY: install train run-backend run-frontend test docker-build docker-up clean help

help:
	@echo "Available commands:"
	@echo "  make install        - Install python dependencies declared in requirements.txt"
	@echo "  make train          - Run the ML model training and evaluation pipeline"
	@echo "  make run-backend    - Spin up the local FastAPI REST server (port 8000)"
	@echo "  make run-frontend   - Launch the local Streamlit dashboard UI (port 8501)"
	@echo "  make test           - Execute the unit, integration, and security test suite"
	@echo "  make docker-build   - Build frontend and backend container images"
	@echo "  make docker-up      - Launch frontend and backend services in detached compose state"
	@echo "  make clean          - Purge bytecode cache, build artifacts, and output files"

install:
	pip install -r requirements.txt

train:
	python -m models.pipeline

run-backend:
	python -m api.main

run-frontend:
	streamlit run frontend/app.py

test:
	pytest -v --cov=.

docker-build:
	docker compose build

docker-up:
	docker compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	@echo "Cleanup completed."
