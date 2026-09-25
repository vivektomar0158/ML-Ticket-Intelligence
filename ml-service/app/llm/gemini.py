"""AI gateway: every LLM call in the system goes through here.

Responsibilities: model rotation across per-model quotas, cooldowns on 429/503, bounded concurrency, timeout,
disk cache (deterministic re-runs are free), token + cost accounting, and a deterministic `mock` provider
so tests / demos run without any API key or quota.
"""
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from app.config import settings


class LlmUnavailable(RuntimeError):
    """No usable LLM right now (no key, quota exhausted, provider down). Callers degrade gracefully."""


class Gateway:
    def __init__(self):
        self.models = [m.strip() for m in settings.gemini_models.split(",") if m.strip()]
        self._sem = asyncio.Semaphore(settings.llm_concurrency)
        self._exhausted: dict[str, float] = {}   # model -> monotonic time it may be retried (daily quota: 1h)
        self._cooldown: dict[str, float] = {}
        self._no_thinking: set[str] = set()
        self._rr = 0
        self._client = None
        self.cache_dir = Path(settings.llm_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"calls": 0, "cache_hits": 0, "in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0, "errors": 0}

    # ---------------------------------------------------------------- public
    async def generate_json(self, prompt: str, schema, *, version: str, temperature: float = 0.0, retries: int = 12,
                            use_cache: bool = True) -> tuple[dict, dict]:
        """Return (parsed JSON, meta). meta = {model, latency_ms, input_tokens, output_tokens, cost_usd, cached}."""
        key = hashlib.sha256(f"{version}|{temperature}|{prompt}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if use_cache and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            self.stats["cache_hits"] += 1
            return d["data"], {**d["meta"], "cached": True}

        if settings.llm_provider == "mock":
            data, meta = _mock(prompt, version), {"model": "mock", "latency_ms": 1, "input_tokens": 0, "output_tokens": 0,
                                                  "cost_usd": 0.0}
        else:
            if not settings.gemini_api_key:
                raise LlmUnavailable("GEMINI_API_KEY is not configured")
            data, meta = await self._call(prompt, schema, temperature, retries)
        if use_cache:
            path.write_text(json.dumps({"data": data, "meta": meta}), encoding="utf-8")
        return data, {**meta, "cached": False}

    # --------------------------------------------------------------- internals
    def _client_(self):
        if self._client is None:
            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    async def _pick(self) -> str:
        while True:
            now = time.monotonic()
            live = [m for m in self.models if self._exhausted.get(m, 0) <= now]
            if not live:
                raise LlmUnavailable(f"quota exhausted for all models {self.models}")
            for k in range(len(live)):
                m = live[(self._rr + k) % len(live)]
                if self._cooldown.get(m, 0) <= now:
                    self._rr = (self._rr + k + 1) % len(live)
                    return m
            await asyncio.sleep(max(min(self._cooldown[m] for m in live) - now, 0.25))

    async def _call(self, prompt, schema, temperature, retries):
        last = None
        async with self._sem:
            for attempt in range(retries):
                model = await self._pick()
                kw = dict(response_mime_type="application/json", response_schema=schema, temperature=temperature)
                if model not in self._no_thinking:
                    kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                t = time.perf_counter()
                try:
                    r = await asyncio.wait_for(
                        self._client_().aio.models.generate_content(
                            model=model, contents=prompt, config=types.GenerateContentConfig(**kw)),
                        timeout=settings.llm_timeout_s)
                    data = json.loads(r.text)
                    u = r.usage_metadata
                    tin, tout = u.prompt_token_count or 0, u.candidates_token_count or 0
                    cost = tin * settings.price_in_per_m / 1e6 + tout * settings.price_out_per_m / 1e6
                    s = self.stats
                    s["calls"] += 1
                    s["in_tokens"] += tin
                    s["out_tokens"] += tout
                    s["cost_usd"] += cost
                    return data, {"model": model, "latency_ms": int((time.perf_counter() - t) * 1000),
                                  "input_tokens": tin, "output_tokens": tout, "cost_usd": cost}
                except LlmUnavailable:
                    raise
                except Exception as e:  # noqa: BLE001 - provider errors are heterogeneous
                    last = e
                    self.stats["errors"] += 1
                    s = str(e)
                    if "429" in s and "PerDay" in s:
                        self._exhausted[model] = time.monotonic() + 3600
                    elif "429" in s:
                        m = re.search(r"retry in ([\d.]+)s", s)
                        self._cooldown[model] = time.monotonic() + (float(m.group(1)) if m else 20) + 1
                    elif "400" in s and "INVALID_ARGUMENT" in s:
                        if model in self._no_thinking:
                            self._exhausted[model] = time.monotonic() + 3600
                        self._no_thinking.add(model)
                    elif isinstance(e, json.JSONDecodeError) or isinstance(e, asyncio.TimeoutError) or "503" in s or "500" in s:
                        await asyncio.sleep(min(2 ** min(attempt, 4), 16))
                    else:
                        await asyncio.sleep(1)
        raise LlmUnavailable(f"LLM call failed after {retries} attempts: {str(last)[:200]}")


def _mock(prompt: str, version: str) -> dict:
    """Deterministic offline stand-in. Keyword rules for classify; grounded template for drafts."""
    if version.startswith("classify"):
        t = prompt.split("<ticket>")[-1].lower()
        rules = [("BILLING", ["invoice", "charge", "refund", "billing", "payment", "vat", "plan"]),
                 ("LOGIN_ACCESS", ["log in", "login", "sign in", "password", "sso", "403", "401", "mfa", "locked"]),
                 ("PERFORMANCE", ["slow", "timeout", "latency", "sync is", "loading"]),
                 ("INTEGRATION", ["api", "webhook", "slack", "jira", "zapier", "token"]),
                 ("DATA_PRIVACY", ["gdpr", "export my data", "breach", "audit log", "retention"]),
                 ("ACCOUNT_MANAGEMENT", ["seat", "role", "permission", "owner", "workspace"]),
                 ("FEATURE_REQUEST", ["would be great", "feature", "please add", "support for"])]
        cat = next((c for c, ks in rules if any(k in t for k in ks)), "BUG")
        pr = "URGENT" if any(k in t for k in ["asap", "outage", "everyone", "down"]) else "MEDIUM"
        return {"category": cat, "priority": pr, "confidence": 0.7, "rationale": "mock keyword rules"}
    if version.startswith("draft"):
        ids = [int(x) for x in re.findall(r"\[T-(\d+)\] Subject", prompt)]
        first = re.search(r"\[T-\d+\] Subject:.*?\nProblem:.*?\nResolution: (.*?)(?:\n\n|\Z)", prompt, re.S)
        if ids and first:
            return {"reply": f"Hi, thanks for reaching out and sorry for the trouble. {first.group(1).strip()} [T-{ids[0]}] "
                             "Let us know if that does not resolve it.",
                    "citations": [{"ticketId": ids[0], "why": "closest resolved ticket"}], "confidence": 0.7, "needsInfo": []}
        return {"reply": "Hi, thanks for reaching out and sorry for the trouble. We are investigating; could you share "
                         "when this started and any error message you see?",
                "citations": [], "confidence": 0.2, "needsInfo": ["When did this start?", "Any error message?"]}
    return {"groundedness": 3, "correctness": 3, "completeness": 3, "tone": 4, "rationale": "mock judge"}


gateway = Gateway()
