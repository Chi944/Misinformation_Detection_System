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

