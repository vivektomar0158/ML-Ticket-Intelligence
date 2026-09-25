import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.llm import prompts
from app.llm.gemini import LlmUnavailable, gateway
from app.llm.pii import redact
from app.models.registry import EMB_VERSION, Registry, registry
from app.schemas import (AnalyzeItem, AnalyzeRequest, AnalyzeResponse, CitationOut, DraftRequest, DraftResponse,
                         EmbedRequest, EmbedResponse, EscalationItem, EscalationRequest, EscalationResponse, JudgeRequest,
                         JudgeResponse, LlmClassifyRequest, LlmClassifyResponse, LlmMeta)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("ml")

LLM_CALLS = Counter("llm_calls_total", "LLM calls", ["task", "model", "cached"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["kind"])
LLM_COST = Counter("llm_cost_usd_total", "Estimated LLM spend in USD")
LLM_LATENCY = Histogram("llm_latency_seconds", "LLM call latency", ["task"])
INFER_LATENCY = Histogram("inference_seconds", "Model inference latency", ["task"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    await run_in_threadpool(registry.load)
    log.info("models loaded in %.1fs: %s", time.perf_counter() - t, registry.versions)
    yield


app = FastAPI(title="Ticket Intelligence ML Service", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.exception_handler(LlmUnavailable)
async def llm_unavailable(_: Request, exc: LlmUnavailable):
    # Callers (the Spring workers) treat 503 + code as "retry later / degrade", never as a bug.
    return JSONResponse(status_code=503, content={"code": "LLM_UNAVAILABLE", "detail": str(exc)[:300]})


# ------------------------------------------------------------------ health
@app.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    if not registry.loaded:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "models": registry.versions, "llmProvider": settings.llm_provider,
            "geminiConfigured": bool(settings.gemini_api_key)}


@app.get("/v1/models")
def models() -> dict:
    return {"versions": registry.versions, "confidenceThreshold": settings.confidence_threshold,
            "seniorThreshold": registry.senior_threshold, "llmModels": gateway.models, "llmStats": gateway.stats}


# ---------------------------------------------------------- model lifecycle
@app.post("/admin/reload")
async def reload_models(x_admin_key: str | None = Header(default=None)):
    """Hot-reload artifacts after a promoted retrain (no restart, no dropped requests: the swap happens after loading)."""
    if x_admin_key != settings.admin_key:
        raise HTTPException(status_code=403, detail="forbidden")
    fresh = Registry()
    await run_in_threadpool(fresh.load)
    fresh._cache = registry._cache            # keep warm embeddings (embedding model unchanged)
    registry.__dict__.update({k: v for k, v in fresh.__dict__.items() if k != "_lock"})
    log.info("models reloaded: %s", registry.versions)
    return {"versions": registry.versions}


# ------------------------------------------------------------- fast models
@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    t = time.perf_counter()
    res = await run_in_threadpool(registry.analyze, [x.model_dump() for x in req.tickets])
    INFER_LATENCY.labels("analyze").observe(time.perf_counter() - t)
    items = [AnalyzeItem(ticketId=tk.id, embedding=[float(v) for v in r["embedding"]] if req.includeEmbedding else None,
                         embeddingModel=EMB_VERSION, category=r["category"], priority=r["priority"], needsLlm=r["needsLlm"])
             for tk, r in zip(req.tickets, res)]
    return AnalyzeResponse(items=items, latencyMs=(time.perf_counter() - t) * 1000)


@app.post("/v1/escalation", response_model=EscalationResponse)
async def escalation(req: EscalationRequest):
    t = time.perf_counter()
    res = await run_in_threadpool(registry.escalation, [x.model_dump() for x in req.tickets])
    INFER_LATENCY.labels("escalation").observe(time.perf_counter() - t)
    return EscalationResponse(items=[EscalationItem(ticketId=tk.id, **r) for tk, r in zip(req.tickets, res)],
                              seniorThreshold=registry.senior_threshold)


@app.post("/v1/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    v = await run_in_threadpool(registry.embed, req.texts)
    return EmbedResponse(embeddings=[[float(x) for x in row] for row in v], model=EMB_VERSION)


# ---------------------------------------------------------------- LLM tasks
def _meta(m: dict, version: str, task: str) -> LlmMeta:
    LLM_CALLS.labels(task, m["model"], str(m["cached"]).lower()).inc()
    if not m["cached"]:
        LLM_TOKENS.labels("input").inc(m["input_tokens"])
        LLM_TOKENS.labels("output").inc(m["output_tokens"])
        LLM_COST.inc(m["cost_usd"])
        LLM_LATENCY.labels(task).observe(m["latency_ms"] / 1000)
    return LlmMeta(model=m["model"], latencyMs=m["latency_ms"], inputTokens=m["input_tokens"], outputTokens=m["output_tokens"],
                   costUsd=m["cost_usd"], cached=m["cached"], promptVersion=version)


@app.post("/v1/llm/classify", response_model=LlmClassifyResponse)
async def llm_classify(req: LlmClassifyRequest):
    text, _ = redact(f"{req.subject}\n{req.body}")
    d, m = await gateway.generate_json(prompts.classify_prompt(text), prompts.ClassifyOut, version=prompts.CLASSIFY_VERSION)
    cat = d["category"] if d["category"] in prompts.CATEGORIES else "BUG"
    pri = d["priority"] if d["priority"] in prompts.PRIORITIES else "MEDIUM"
    return LlmClassifyResponse(category=cat, priority=pri, confidence=float(np.clip(d["confidence"], 0, 1)),
                               rationale=d["rationale"][:300], meta=_meta(m, prompts.CLASSIFY_VERSION, "classify"))


def grounding_of(sources) -> str:
    if not sources:
        return "NONE"
    best = max(s.score for s in sources)
    return "STRONG" if best >= 0.75 else "WEAK" if best >= 0.55 else "NONE"


@app.post("/v1/llm/draft", response_model=DraftResponse)
async def llm_draft(req: DraftRequest):
    grounding = grounding_of(req.sources)
    # NONE -> the prompt switches to "acknowledge + ask questions"; irrelevant sources are not shown to the model
    sources = req.sources if grounding != "NONE" else []
    t, _ = redact(req.ticket.subject)
    b, _ = redact(req.ticket.body)
    tk = {**req.ticket.model_dump(), "subject": t, "body": b}
    src = [{"id": s.id, "subject": s.subject, "problem": redact(s.problem)[0], "resolution": s.resolution} for s in sources]
    d, m = await gateway.generate_json(prompts.draft_prompt(tk, src, req.instruction), prompts.DraftOut,
                                       version=prompts.DRAFT_VERSION, temperature=0.2)
    valid = {s["id"] for s in src}
    cits = [CitationOut(ticketId=c["ticketId"], why=c["why"][:200]) for c in d.get("citations", []) if c["ticketId"] in valid]
    stripped = len(d.get("citations", [])) - len(cits)
    reply = d["reply"].strip()
    # inline [T-id] tags for ids we did not provide are hallucinations too
    bad_tags = {int(x) for x in re.findall(r"\[T-(\d+)\]", reply)} - valid
    for x in bad_tags:
        reply = reply.replace(f"[T-{x}]", "")
    stripped += len(bad_tags)
    conf = float(np.clip(d.get("confidence", 0.5), 0, 1))
    if stripped:
        conf = min(conf, 0.5)
        if grounding == "STRONG":
            grounding = "WEAK"
    if len(reply) < 20:
        raise HTTPException(status_code=502, detail={"code": "BAD_DRAFT", "detail": "draft too short"})
    return DraftResponse(reply=reply[:4000], citations=cits, confidence=conf, needsInfo=[q[:200] for q in d.get("needsInfo", [])][:5],
                         grounding=grounding, strippedCitations=stripped, meta=_meta(m, prompts.DRAFT_VERSION, "draft"))


@app.post("/v1/llm/judge", response_model=JudgeResponse)
async def llm_judge(req: JudgeRequest):
    src = [{"id": s.id, "resolution": s.resolution} for s in req.sources]
    d, m = await gateway.generate_json(prompts.judge_prompt(req.ticket.model_dump(), req.draft, src, req.goldFix),
                                       prompts.JudgeOut, version=prompts.JUDGE_VERSION)
    clip = lambda x: int(np.clip(x, 1, 5))
    return JudgeResponse(groundedness=clip(d["groundedness"]), correctness=clip(d["correctness"]),
                         completeness=clip(d["completeness"]), tone=clip(d["tone"]), rationale=d["rationale"][:300],
                         meta=_meta(m, prompts.JUDGE_VERSION, "judge"))
