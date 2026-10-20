FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY lora_serve ./lora_serve
RUN pip install ".[ml]"

# mount or copy the merged model to /app/artifacts/merged (see README)
ENV PREDICTOR=hf MODEL_DIR=/app/artifacts/merged INT8=1
EXPOSE 8000
CMD ["uvicorn", "lora_serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
