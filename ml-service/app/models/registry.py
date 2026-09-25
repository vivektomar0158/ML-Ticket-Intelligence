"""Loads model artifacts once at startup and exposes batch inference. Everything here is CPU-only and thread-safe
(read-only models); the FastAPI layer runs it in a threadpool so the event loop stays free for LLM I/O."""
import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np

from app.config import settings
from app.features.text import extract, extract_matrix
from app.models.priority import build

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_VERSION = "minilm-l6-v2@1"
PRIORITIES = ["LOW", "MEDIUM", "HIGH", "URGENT"]
PRODUCTS = ["web-app", "mobile-app", "api", "desktop-sync", "admin-console"]
TIER_ORD = {"FREE": 0, "PRO": 1, "ENTERPRISE": 2}


def _current(task: str) -> Path:
    base = Path(settings.artifacts_dir) / task
    return base / (base / "CURRENT").read_text().strip()


class Registry:
    def __init__(self):
        self.loaded = False
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()   # content-hash -> embedding (LRU)
        self._lock = threading.Lock()
        self.versions: dict[str, str] = {}

    def load(self):
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(EMB_MODEL)
        d = _current("category")
        self.cat = joblib.load(d / "emb_lr.joblib")
        self.cat_classes = list(self.cat.classes_)
        self.versions["category"] = f"emb_lr@{d.name}"
        d = _current("priority")
        p = joblib.load(d / "priority.joblib")
        self.pri_feat, self.pri_scaler, self.pri_lr, self.urgent_w = p["feat"], p["scaler"], p["lr"], p["urgent_weight"]
        self.versions["priority"] = f"priority_lr@{d.name}"
        d = _current("escalation")
        self.esc = joblib.load(d / "escalation.joblib")
        self.senior_threshold = json.loads((d / "meta.json").read_text()).get("threshold_senior", 0.5)
        self.versions["escalation"] = f"lgbm_platt@{d.name}"
        self.versions["embedding"] = EMB_VERSION
        self.loaded = True

    # ------------------------------------------------------------ embeddings
    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        with self._lock:
            miss = [i for i, k in enumerate(keys) if k not in self._cache]
        if miss:
            vecs = self.embedder.encode([texts[i] for i in miss], batch_size=32, normalize_embeddings=True,
                                        show_progress_bar=False)
            with self._lock:
                for i, v in zip(miss, vecs):
                    self._cache[keys[i]] = v.astype(np.float32)
                    if len(self._cache) > 5000:
                        self._cache.popitem(last=False)
        with self._lock:
            return np.vstack([self._cache[k] for k in keys])

    # -------------------------------------------------------------- analyze
    def analyze(self, tickets: list[dict]) -> list[dict]:
        texts = [f"{t['subject']}\n{t['body']}" for t in tickets]
        E = self.embed(texts)
        Pc = self.cat.predict_proba(E)
        Xp = build(self.pri_feat, self.pri_scaler, texts, [t["tier"] for t in tickets])
        Pp = self.pri_lr.predict_proba(Xp)
        Pp_adj = Pp * np.array([1, 1, 1, self.urgent_w])
        Pp_adj = Pp_adj / Pp_adj.sum(1, keepdims=True)
        out = []
        for i, txt in enumerate(texts):
            ci, pi = int(Pc[i].argmax()), int(Pp_adj[i].argmax())
            out.append({
                "embedding": E[i],
                "category": {"label": self.cat_classes[ci], "confidence": float(Pc[i][ci]),
                             "model": self.versions["category"],
                             "probs": {c: float(p) for c, p in zip(self.cat_classes, Pc[i])}},
                "priority": {"label": PRIORITIES[pi], "confidence": float(Pp_adj[i][pi]),
                             "model": self.versions["priority"], "signals": extract(txt)[1],
                             "probs": {p: float(v) for p, v in zip(PRIORITIES, Pp_adj[i])}},
                "needsLlm": bool(Pc[i][ci] < settings.confidence_threshold)})
        return out

    # ----------------------------------------------------------- escalation
    def escalation(self, tickets: list[dict]) -> list[dict]:
        e = self.esc
        texts = [f"{t['subject']}\n{t['body']}" for t in tickets]
        E = np.vstack([np.asarray(t["embedding"], np.float32) if t.get("embedding") else self.embed([x])[0]
                       for t, x in zip(tickets, texts)])
        S = extract_matrix(texts)
        Pc = e["category_lr"].predict_proba(E)
        pr = e["priority"]
        Pp = pr["lr"].predict_proba(build(pr["feat"], pr["scaler"], texts, [t["tier"] for t in tickets]))
        meta = np.array([[TIER_ORD.get(t["tier"], 0), PRODUCTS.index(t["product"]) if t["product"] in PRODUCTS else 0,
                          t["hour"], t["weekday"], int(t["hour"] < 9 or t["hour"] >= 18 or t["weekday"] >= 5),
                          t["priorTickets7d"], t["similarRecent"]] for t in tickets], np.float32)
        X = np.hstack([S, e["pca"].transform(E), Pc, Pp, meta]).astype(np.float32)
        raw = e["gbm"].predict_proba(X)[:, 1]
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        risk = e["platt"].predict_proba(np.log(clipped / (1 - clipped)).reshape(-1, 1))[:, 1]
        contrib = e["gbm"].booster_.predict(X, pred_contrib=True)[:, :-1]
        names = e["names"]
        out = []
        for i in range(len(tickets)):
            agg: dict[str, float] = {}
            for n, c in zip(names, contrib[i]):
                key = "text_embedding" if n.startswith("emb_pc") else n
                agg[key] = agg.get(key, 0.0) + float(c)
            top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:3]
            out.append({"risk": float(risk[i]), "rawScore": float(raw[i]),
                        "route": "SENIOR" if risk[i] >= self.senior_threshold else "STANDARD",
                        "topFactors": [{"feature": k, "contribution": v} for k, v in top],
                        "model": self.versions["escalation"]})
        return out


registry = Registry()
