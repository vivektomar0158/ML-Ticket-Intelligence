"""Shared helpers for training / evaluation scripts."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
ART = ROOT / "ml-service" / "artifacts"
METRICS = ROOT / "eval" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_VERSION = "minilm-l6-v2@1"
CATEGORIES = ["BILLING", "LOGIN_ACCESS", "BUG", "FEATURE_REQUEST", "PERFORMANCE", "INTEGRATION",
              "ACCOUNT_MANAGEMENT", "DATA_PRIVACY"]
PRIORITIES = ["LOW", "MEDIUM", "HIGH", "URGENT"]

_model = None


def embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMB_MODEL)
    return _model


def embed(texts, batch_size=64) -> np.ndarray:
    return embedder().encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=False).astype(np.float32)


def load_tickets() -> tuple[pd.DataFrame, np.ndarray]:
    """All tickets + aligned MiniLM embeddings (cached on disk)."""
    df = pd.read_parquet(DATA / "tickets.parquet").reset_index(drop=True)
    cache = DATA / "emb_minilm.npy"
    if cache.exists() and np.load(cache).shape[0] == len(df):
        return df, np.load(cache)
    e = embed(df.text.tolist())
    np.save(cache, e)
    return df, e


def epoch_s(s: pd.Series) -> np.ndarray:
    """Seconds since epoch, independent of the datetime unit pandas chose (us/ns) and of tz-awareness."""
    s = pd.to_datetime(s, utc=True)
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()).to_numpy().astype("int64")


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def save_metrics(name: str, obj: dict) -> None:
    (METRICS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_j), encoding="utf-8")


def load_metrics(name: str) -> dict:
    p = METRICS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _j(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def latency_ms(fn, items, batch: int = 1, repeats: int = 200) -> dict:
    """Per-call latency of fn(batch_of_items); warm, batch size `batch`."""
    items = list(items)
    fn(items[:batch])  # warm-up
    ts = []
    for i in range(min(repeats, len(items) // batch)):
        chunk = items[i * batch:(i + 1) * batch]
        t = time.perf_counter()
        fn(chunk)
        ts.append((time.perf_counter() - t) * 1000)
    return {"batch": batch, "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "per_ticket_ms": float(np.mean(ts) / batch)}


def ece(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error for binary probabilities."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(e)


def multiclass_ece(y_idx: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(1)
    correct = (proba.argmax(1) == y_idx).astype(float)
    return ece(correct, conf, bins)
