PY ?= python

.PHONY: install lint test train eval export plot serve

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[ml,dev]"

lint:
	ruff check .

test:
	pytest -q

train:
	$(PY) -m lora_serve.train

eval:
	$(PY) -m lora_serve.evaluate

export:
	$(PY) -m lora_serve.export

plot:
	$(PY) scripts/plot_results.py

serve:
	PREDICTOR=hf uvicorn lora_serve.app:app --port 8000
