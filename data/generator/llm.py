"""Thin Gemini wrapper for data generation: JSON output, disk cache, retries, model rotation.

Free-tier keys have a small per-model *daily* request quota (e.g. 20/day for gemini-2.5-flash), so
generation rotates through GEN_MODELS: when a model's daily quota is exhausted it is skipped and the
next one is used. Results are cached on disk keyed by prompt only (not model), so runs resume for free.
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
CACHE = ROOT / "data" / "raw" / "llm_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# cheapest / highest-quota first
GEN_MODELS = os.environ.get(
    "GEMINI_GEN_MODELS",
    "gemini-3.1-flash-lite,gemini-2.5-flash-lite,gemini-3.5-flash-lite,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash,gemini-2.5-flash",
).split(",")
_KEY_MODEL = "gemini-2.5-flash"  # frozen so cache entries written before rotation stay valid

_client = None
_lock = threading.Lock()
_exhausted: set[str] = set()      # daily quota gone
_cooldown: dict[str, float] = {}  # per-minute quota hit: model usable again after this monotonic time
_rr = 0                           # round-robin cursor
_no_thinking_cfg: set[str] = set()  # models that reject thinking_budget=0


class AllModelsExhausted(RuntimeError):
    pass


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _pick_model() -> str:
    """Round-robin over models that are neither daily-exhausted nor cooling down; wait if all cooling."""
    global _rr
    while True:
        with _lock:
            live = [m for m in GEN_MODELS if m not in _exhausted]
            if not live:
                raise AllModelsExhausted(f"daily quota exhausted for all of {GEN_MODELS}; re-run later (cache resumes)")
            now = time.monotonic()
            for k in range(len(live)):
                m = live[(_rr + k) % len(live)]
                if _cooldown.get(m, 0) <= now:
                    _rr = (_rr + k + 1) % len(live)
                    return m
            wait = min(_cooldown[m] for m in live) - now
        time.sleep(max(wait, 0.5))


def generate_json(prompt: str, schema=None, temperature: float = 0.9, tag: str = "gen", retries: int = 40):
    """Return parsed JSON. Cached on disk by (prompt, temperature, tag)."""
    key = hashlib.sha256(f"{_KEY_MODEL}|{temperature}|{tag}|{prompt}".encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["data"]

    for attempt in range(retries):
        model = _pick_model()
        kwargs = dict(response_mime_type="application/json", response_schema=schema, temperature=temperature)
        if model not in _no_thinking_cfg:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        try:
            t = time.perf_counter()
            r = client().models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**kwargs))
            data = json.loads(r.text)
            u = r.usage_metadata
            path.write_text(json.dumps({
                "data": data, "model": model, "latency_s": round(time.perf_counter() - t, 2),
                "in_tokens": u.prompt_token_count, "out_tokens": u.candidates_token_count,
            }), encoding="utf-8")
            return data
        except Exception as e:
            s = str(e)
            if "429" in s and "PerDay" in s:
                with _lock:
                    if model not in _exhausted:
                        print(f"  !! daily quota exhausted for {model}; rotating")
                    _exhausted.add(model)
                continue
            if "429" in s and "PerMinute" in s:
                m = re.search(r"retry in ([\d.]+)s", s)
                with _lock:
                    _cooldown[model] = time.monotonic() + (float(m.group(1)) if m else 30) + 1
                continue
            if "400" in s and "INVALID_ARGUMENT" in s:
                with _lock:
                    if model not in _no_thinking_cfg and "thinking_config" in kwargs:
                        _no_thinking_cfg.add(model)   # first 400: retry without thinking_budget
                    else:
                        print(f"  !! {model} rejects requests ({s[:90]!r}); dropping it")
                        _exhausted.add(model)
                continue
            wait = min(60, 2 ** attempt * 2)
            print(f"  [{tag}] {model} attempt {attempt + 1} failed: {s[:100]!r} -> retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"generation failed after {retries} attempts: {tag}")


def usage_totals() -> dict:
    tin = tout = n = 0
    by_model: dict[str, int] = {}
    for f in CACHE.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        tin += d.get("in_tokens") or 0
        tout += d.get("out_tokens") or 0
        by_model[d.get("model", "?")] = by_model.get(d.get("model", "?"), 0) + 1
        n += 1
    return {"calls": n, "in_tokens": tin, "out_tokens": tout, "by_model": by_model}
