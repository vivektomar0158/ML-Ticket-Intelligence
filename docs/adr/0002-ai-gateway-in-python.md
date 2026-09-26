# ADR 0002: All LLM calls go through the Python ML service

**Decision.** Spring Boot never calls Gemini. The ML service exposes `/v1/llm/{classify,draft,judge}`.

**Why.** Prompts, JSON schemas, PII redaction, caching, token/cost accounting and model rotation (needed on a free-tier key) live in one place, and the offline evaluation scripts import the exact same gateway, so evaluation measures production behaviour. Spring stays an orchestrator and the sole owner of the database.

**Consequences.** One extra hop; the API treats `503 LLM_UNAVAILABLE` as "wait", not "fail" (no circuit-breaker trip, no burned retries).
