# ADR 0001: Postgres job table with SKIP LOCKED instead of a message broker

**Decision.** Background work (TRIAGE, LLM_CLASSIFY, DRAFT) lives in a `jobs` table claimed with `FOR UPDATE SKIP LOCKED`.

**Why.** The ticket row and its job are inserted in the same transaction (a free transactional outbox: no lost or phantom jobs), and there is one less system to run. Measured: 500-ticket burst triaged in 10.8 s; 300 jobs claimed exactly once by 8 concurrent workers (integration test).

**Consequences.** Retry with backoff, a reaper (visibility timeout 5 min) and a dead-letter state are our code. Outages of a dependency *defer* jobs without consuming attempts. If job rates outgrow Postgres, job types map cleanly onto broker queues.
