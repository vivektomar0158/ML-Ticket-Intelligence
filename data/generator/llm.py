"""Thin Gemini wrapper for data generation: JSON output, disk cache, retries, cost log."""
import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
CACHE = ROOT / "data" / "raw" / "llm_cache"
CACHE.mkdir(parents=True, exist_ok=True)
GEN_MODEL = os.environ.get("GEMINI_GEN_MODEL", "gemini-2.5-flash")

_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def generate_json(prompt: str, schema=None, temperature: float = 0.9, tag: str = "gen", retries: int = 5):
    """Return parsed JSON. Cached on disk by (model, prompt, temperature, tag)."""
    key = hashlib.sha256(f"{GEN_MODEL}|{temperature}|{tag}|{prompt}".encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["data"]

    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(retries):
        try:
            t = time.perf_counter()
            r = client().models.generate_content(model=GEN_MODEL, contents=prompt, config=cfg)
            data = json.loads(r.text)
            u = r.usage_metadata
            path.write_text(json.dumps({
                "data": data, "model": GEN_MODEL, "latency_s": round(time.perf_counter() - t, 2),
                "in_tokens": u.prompt_token_count, "out_tokens": u.candidates_token_count,
            }), encoding="utf-8")
            return data
        except Exception as e:  # rate limits, transient 5xx, bad JSON
            wait = min(60, 2 ** attempt * 2)
            print(f"  [{tag}] attempt {attempt + 1} failed: {str(e)[:120]} -> retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"generation failed after {retries} attempts: {tag}")


def usage_totals() -> dict:
    tin = tout = n = 0
    for f in CACHE.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        tin += d.get("in_tokens") or 0
        tout += d.get("out_tokens") or 0
        n += 1
    return {"calls": n, "in_tokens": tin, "out_tokens": tout}
