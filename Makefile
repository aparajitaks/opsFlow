# ==========================================
# Makefile - opsFlow Automation Command Hooks
# ==========================================

.PHONY: install train evaluate test clean help

help:
	@echo "Available commands:"
	@echo "  make install        - Install python dependencies declared in requirements.txt"
	@echo "  make train          - Execute Task 3 ML model training pipeline"
	@echo "  make evaluate       - Run Holdout ML evaluations and generate plots"
	@echo "  make test           - Execute the python unit and integration test suites"
	@echo "  make clean          - Purge bytecode cache, build artifacts, and logs"

install:
	pip install -r requirements.txt

train:
	python main.py --train

evaluate:
	python main.py --evaluate

test:
	pytest -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf models/artifacts/data_splits.pkl
	rm -rf models/artifacts/*.pkl
	rm -rf models/artifacts/plots/*.png
	@echo "Cleanup completed."
