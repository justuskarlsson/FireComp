"""
viewer/backend — FastAPI server for the FireComp research viewer.

Serves index metadata and matplotlib overlay PNGs. No PyTorch.

    uvicorn viewer.backend.main:app --reload --port 8000
"""

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .render import (
    LAYER_INFO, PADDING, SampleArrays, load_pred, render_layer, render_legend,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "viewer"

app = FastAPI(title="FireComp viewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_index() -> dict:
    path = DATA / "index.json"
    if not path.exists():
        raise HTTPException(
            503, f"{path} missing — run firecomp.viewer.export collect")
    import json
    return json.loads(path.read_text())


@app.get("/api/index")
def api_index():
    idx = load_index()
    return {
        "generated_at": idx.get("generated_at"),
        "split": idx.get("split"),
        "fire_type": idx.get("fire_type"),
        "target": idx.get("target"),
        "models": idx.get("models", []),
        "n_samples": idx.get("n_samples"),
        "n_events": idx.get("n_events"),
        "events": idx.get("events", []),
        "samples": idx.get("samples", []),
        "padding": PADDING,
        "layers": [
            {"id": k, **v} for k, v in LAYER_INFO.items()
        ],
    }


@app.get("/api/models")
def api_models():
    return load_index().get("models", [])


@app.get("/api/events/{event_id}")
def api_event(event_id: int):
    idx = load_index()
    for e in idx["events"]:
        if e["id"] == event_id:
            samples = [s for s in idx["samples"] if s["id"] in e["sample_ids"]]
            return {**e, "samples": samples}
    raise HTTPException(404, f"event {event_id} not found")


@app.get("/api/samples/{sample_id}")
def api_sample(sample_id: str):
    idx = load_index()
    for s in idx["samples"]:
        if s["id"] == sample_id:
            return s
    raise HTTPException(404, f"sample {sample_id} not found")


@app.get("/api/samples/{sample_id}/layers/{name}.png")
def api_layer(sample_id: str, name: str,
              model: str = Query(""),
              size: int = Query(512)):
    if name not in LAYER_INFO:
        raise HTTPException(404, f"unknown layer {name}")
    npz = DATA / "samples" / f"{sample_id}.npz"
    if not npz.exists():
        raise HTTPException(404, f"no arrays for {sample_id}")
    arrays = SampleArrays(npz)
    pred = None
    threshold = 0.5
    needs_pred = name in ("prob", "confusion")
    if needs_pred:
        if not model:
            raise HTTPException(400, "model query param required")
        pred_path = DATA / "preds" / model / f"{sample_id}.npy"
        if not pred_path.exists():
            raise HTTPException(404, f"no prediction {model}/{sample_id}")
        pred = load_pred(pred_path)
        threshold = _model_threshold(model)
    png = render_layer(name, arrays, pred, threshold, size=size)
    return Response(
        content=png, media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/legend/{name}.png")
def api_legend(name: str):
    if name not in LAYER_INFO:
        raise HTTPException(404, f"unknown layer {name}")
    return Response(content=render_legend(name), media_type="image/png")


def _model_threshold(model_id: str) -> float:
    for m in load_index().get("models", []):
        if m["id"] == model_id:
            return float(m.get("threshold", 0.5))
    return 0.5
