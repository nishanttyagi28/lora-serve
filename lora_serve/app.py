"""FastAPI service: /predict, /health, /metrics."""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from .guardrails import check_input
from .predictor import LabelError, Predictor, from_env, predict_with_retry

REQUESTS = Counter("predict_requests_total", "Predict requests", ["status"])
BLOCKED = Counter("guardrail_blocks_total", "Requests blocked by guardrails", ["reason"])
RETRIES = Counter("schema_retries_total", "Extra model calls after invalid JSON output")
LABELS = Counter("predicted_label_total", "Predicted labels", ["label"])
LATENCY = Histogram(
    "predict_latency_seconds", "End to end predict latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


class PredictIn(BaseModel):
    text: str


class PredictOut(BaseModel):
    label: str
    attempts: int
    model: str
    latency_ms: float


def create_app(predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(title="lora-serve")
    state = {"p": predictor}

    def get() -> Predictor:
        if state["p"] is None:
            state["p"] = from_env()
        return state["p"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model": get().name}

    @app.post("/predict", response_model=PredictOut)
    def predict(body: PredictIn):
        t0 = time.perf_counter()
        v = check_input(body.text)
        if not v.ok:
            BLOCKED.labels(reason=v.reason).inc()
            REQUESTS.labels(status="blocked").inc()
            raise HTTPException(status_code=422, detail=v.reason)
        try:
            label, attempts = predict_with_retry(get(), body.text)
        except LabelError as e:
            REQUESTS.labels(status="invalid_output").inc()
            raise HTTPException(status_code=502, detail=str(e)) from e
        RETRIES.inc(attempts - 1)
        LABELS.labels(label=label).inc()
        REQUESTS.labels(status="ok").inc()
        dt = time.perf_counter() - t0
        LATENCY.observe(dt)
        return PredictOut(label=label, attempts=attempts, model=get().name, latency_ms=dt * 1000)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
