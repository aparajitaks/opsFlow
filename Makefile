# ==========================================
# Makefile - opsFlow Automation Command Hooks
# ==========================================

.PHONY: install train run test clean help

help:
	@echo "Available commands:"
	@echo "  make install        - Install python dependencies declared in requirements.txt"
	@echo "  make train          - Run the ML model training and evaluation pipeline"
	@echo "  make run            - Launch the local self-contained Streamlit dashboard UI (port 8501)"
	@echo "  make test           - Execute the unit, integration, and security test suite"
	@echo "  make clean          - Purge bytecode cache, build artifacts, and output files"

install:
	pip install -r requirements.txt

train:
	python -m models.pipeline

run:
	streamlit run streamlit_app.py

test:
	pytest -v --cov=.

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	@echo "Cleanup completed."
