# Makefile — opsFlow (Tasks 3 & 4)

.PHONY: install train evaluate test validate query clean help

help:
	@echo "  make install   - pip install -r requirements.txt"
	@echo "  make train     - Task 3: train LR + RF models"
	@echo "  make evaluate  - Task 3: holdout metrics + plots"
	@echo "  make test      - pytest suite"
	@echo "  make validate  - quick dataset load check"
	@echo "  make query Q=  - Task 4: RAG query (requires GROQ_API_KEY in .env)"
	@echo "  make clean     - remove caches and regenerated artifacts"

install:
	./venv/bin/pip install -r requirements.txt

train:
	./venv/bin/python main.py --train

evaluate:
	./venv/bin/python main.py --evaluate

test:
	ML_N_JOBS=1 ./venv/bin/pytest -v

validate:
	./venv/bin/python -c "from models.train import load_dataset; load_dataset(); print('Dataset OK')"

query:
	@test -n "$(Q)" || (echo 'Usage: make query Q="your question"' && exit 1)
	./venv/bin/python main.py --query "$(Q)"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache models/artifacts/data_splits.pkl models/artifacts/*.pkl
	@echo "Cleanup done."
