.PHONY: help install smoke train evaluate test lint format feedback clean

PYTHON  = python
SCRIPTS = scripts
SRC     = src
TESTS   = tests

help:
	@echo "Misinformation Detector - available targets:"
	@echo "  make install     Install all dependencies"
	@echo "  make smoke       Run smoke test (fast pipeline check)"
	@echo "  make train       Train all models on real data"
	@echo "  make train-syn   Train all models on synthetic data"
	@echo "  make evaluate    Run full evaluation pipeline"
	@echo "  make feedback    Run one feedback cycle"
	@echo "  make test        Run all pytest unit tests"
	@echo "  make lint        Run flake8 linter"
	@echo "  make format      Run black formatter"
	@echo "  make clean       Remove generated files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

smoke:
	$(PYTHON) $(SCRIPTS)/smoke_test.py --synthetic

train:
	$(PYTHON) $(SCRIPTS)/train_all.py --data data/train.csv

train-syn:
	$(PYTHON) $(SCRIPTS)/train_all.py --synthetic --n-synthetic 500

evaluate:
	$(PYTHON) $(SCRIPTS)/evaluate.py

evaluate-syn:
	$(PYTHON) $(SCRIPTS)/evaluate.py --synthetic

feedback:
	$(PYTHON) $(SCRIPTS)/run_feedback_cycle.py

test:
	pytest $(TESTS)/ -v --timeout=120

lint:
	flake8 $(SRC)/ $(TESTS)/ $(SCRIPTS)/ --max-line-length=100 --ignore=E501,W503

format:
	black $(SRC)/ $(TESTS)/ $(SCRIPTS)/

download-data:
	$(PYTHON) $(SCRIPTS)/download_sample_data.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f feedback.db RETRAIN_REQUIRED.flag
	rm -rf reports/*.json reports/*.png

.PHONY: all train eval test feedback clean lint format smoke retrain-check

all: train eval test

train:
	python scripts/train_all.py

eval:
	python scripts/evaluate.py

feedback:
	python scripts/run_feedback_cycle.py

test:
	pytest tests/ -v --asyncio-mode=auto

lint:
	flake8 src/ tests/ scripts/ --max-line-length=100

format:
	black src/ tests/ scripts/

smoke:
	timeout 60 python scripts/smoke_test.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -f RETRAIN_REQUIRED.flag

retrain-check:
	@if [ -f RETRAIN_REQUIRED.flag ]; then \
		echo "Retrain required:"; \
		cat RETRAIN_REQUIRED.flag; \
	else \
		echo "No retrain required"; \
	fi

