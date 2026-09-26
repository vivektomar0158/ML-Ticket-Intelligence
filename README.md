# Support Ticket Intelligence Platform

Automates the **first pass** on support tickets: classify, score urgency and escalation risk, spot duplicate reports (outages), retrieve similar resolved tickets, and draft a grounded reply. **A human agent approves every reply**; approvals and edits feed a retraining loop.

```
React dashboard ──► Spring Boot API ──► PostgreSQL + pgvector      (tickets, embeddings, jobs, drafts, feedback)
   (SSE live)            │  job queue (SKIP LOCKED) · dedup · hybrid retrieval · review
                         ├──► Python ML service (FastAPI): classifiers, escalation model, embeddings, AI gateway ──► Gemini
```

| Step | Technique | Where |
|---|---|---|
| Category | MiniLM embeddings + logistic regression (calibrated), Gemini zero-shot for low confidence (cascade) | `ml-service/` |
| Priority | TF-IDF + urgency signals (deadline/outage/error-code) + tier → LR, URGENT threshold tuned for recall | `ml-service/` |
| Escalation risk | LightGBM + Platt calibration, top-3 explanations per ticket | `ml-service/` |
| Duplicates / outages | pgvector kNN + time window + union-find clusters; incident = ≥5 reports, ≥3 customers, 60 min | `api/dedup` |
| Retrieval (RAG) | vector + Postgres full-text, fused with RRF, one hit per duplicate cluster | `api/retrieval` |
| Draft (RAG) | Gemini, JSON output, citations validated against retrieved ids, PII redacted first | `ml-service/app/llm` |
| Async | Postgres job table + `FOR UPDATE SKIP LOCKED`, backoff, reaper, dead-letter | `api/jobs` |

## Results (details and caveats in [`eval/report.md`](eval/report.md))

The data is synthetic, so **read the robustness rows first**.

| | Result |
|---|---|
| Category macro-F1, issues seen in training | 0.97 |
| Category macro-F1, **genuinely new issue types** | **≈ 0.59–0.60** (both cheap models); the headline 0.97 mostly measures recognising known issues |
| Zero-shot Gemini vs trained model (same 56 tickets) | 0.63 vs 1.00 accuracy; the LLM cannot know this taxonomy's boundaries |
| Escalation ROC-AUC / calibration | 0.73 (ceiling with the generator's hidden inputs: 0.80) / ECE 0.033 |
| Outage detection, held-out period, through the running system | **6/6 detected, 0 false alarms**, ~5 min after the first report |
| Retrieval recall@5 / citation precision (end-to-end) | 0.96 / 0.93 |
| 500-ticket outage burst | fully triaged in 10.8 s; ingest p95 10 ms at steady load |
| Chaos: ML service killed; API hard-killed with 54 jobs in flight | 0 tickets lost, 0 duplicate predictions |

**Not measured:** Gemini on the unseen-issue regime and LLM-judge draft scores (free-tier quota), agent approval rate (no real agents). See the report's §11 for every known gap.

## Run it

Prerequisites: Docker, and (for retraining/eval) Python 3.11+ with [`uv`](https://docs.astral.sh/uv/). Java 17 + Node 20 only if you run outside Docker.

```bash
cp .env.example .env            # put GEMINI_API_KEY in it (LLM_PROVIDER=mock in .env runs without any key)
docker compose up --build       # dashboard http://localhost:8081  ·  API :8080  ·  ML :8000  ·  Postgres :5433
docker compose --profile observability up   # + Prometheus :9090, Grafana :3000 (dashboard provisioned)
```

Trained model files live in `ml-service/artifacts/` (git-ignored) and the dataset in `data/processed/`. Build them once (needs a Gemini key for the data step; see *Reproduce*), or the ML service will not start.

Load the RAG history and replay traffic (log in as `agent` / `senior` / `admin`, password = username):

```bash
uv run --project ml-service python scripts/seed.py history              # 6,164 resolved tickets + embeddings
uv run --project ml-service python scripts/seed.py incident --speed 0   # replay a simulated outage
uv run --project ml-service python scripts/seed.py replay --n 300       # ordinary traffic
```

### Local development (no Docker for the apps)

```bash
docker compose up -d postgres
(cd ml-service && uv run uvicorn app.main:app --port 8000)      # LLM_PROVIDER=mock for a key-less run
(cd api && ./mvnw spring-boot:run)
(cd dashboard && npm install && npm run dev)                    # http://localhost:5173 (proxies /api)
```

### Reproduce the numbers

```bash
# 1. data (repo root; Gemini calls are cached in data/raw/llm_cache and resume after quota resets)
uv run --project ml-service python -m data.generator.root_causes
uv run --project ml-service python -m data.generator.sample_specs
uv run --project ml-service python -m data.generator.generate_tickets --workers 6
uv run --project ml-service python -m data.generator.prepare_bitext
uv run --project ml-service python -m data.generator.build_dataset
# 2. models + offline evaluation
cd ml-service
uv run python -m training.train_category && uv run python -m training.train_priority && uv run python -m training.train_escalation
uv run python -m training.eval_robustness && uv run python -m training.eval_dedup_retrieval
uv run python -m training.eval_incidents && uv run python -m training.eval_oracle
uv run python -m training.eval_llm_zeroshot --n 120 && uv run python -m training.eval_cascade   # needs Gemini quota
# 3. end-to-end (stack running, history loaded, test split replayed) + report
cd .. && uv run --project ml-service python eval/pipeline_eval.py && uv run --project ml-service python infra/loadtest.py
uv run --project ml-service python eval/build_report.py
```

## Tests

| Suite | Command | Count |
|---|---|---|
| ML service (models, LLM gateway, guards) | `cd ml-service && uv run pytest` | 18 |
| API integration (real pgvector via Testcontainers + fake ML server) | `cd api && ./mvnw test` | 22 |
| Dashboard unit/component | `cd dashboard && npm test` | 26 |
| Browser E2E against the running stack | `cd dashboard && node e2e/smoke.mjs` | 18 checks |
| Chaos on real processes | `scripts/chaos.sh` | 9 checks |

## Design decisions

Short ADRs are in [`docs/adr`](docs/adr). The ones worth knowing:

- **Postgres is the queue.** A ticket and its job are committed in one transaction, so nothing is lost between them; no broker to run.
- **Two-stage pipeline.** Predictions in < 2 s regardless of LLM latency; the slow draft stage has its own workers, priority ordering, load shedding, and **one draft per outage** instead of one per ticket.
- **Graceful degradation.** LLM down → triage unaffected, drafts wait and resume. ML service down → tickets accepted, jobs deferred without burning retries.
- **Every LLM call goes through the Python AI gateway** (prompts, PII redaction, caching, cost accounting, model rotation), so offline evaluation runs the same code as production.
- **Explainable by default.** Confidence + source (model / LLM / agent) on every label, top escalation factors, cited source tickets on every draft.

## Layout

```
api/          Spring Boot 4 (Java 17): ingest, jobs, triage, dedup, retrieval, drafts, review, metrics, admin
ml-service/   FastAPI service + training/ (models, evaluation, retraining)
dashboard/    React + TypeScript (Vite): queue, ticket review, incidents, upload, metrics
data/         synthetic-data generator, Bitext prep, data card
eval/         evaluation report (report.md), pipeline eval, report builder
infra/        Prometheus / Grafana config, load test
scripts/      seed + replay, reset, chaos
docs/         implementation plan, ADRs, ML service OpenAPI contract
```

## Limitations worth stating

- Tickets are LLM-generated; absolute scores are optimistic (quantified in the report).
- Escalation labels come from a known latent function, so its AUC measures recovery of that function, not real-world behaviour.
- The Gemini key used during development was on the free tier; anything needing more than a few dozen LLM calls per model per day was sampled or left unmeasured.
- Single-node deployment; the scale-out path (multiple API replicas are already safe for workers; schedulers would need ShedLock; a broker only if job rates exceed what Postgres handles) is reasoned, not benchmarked.
