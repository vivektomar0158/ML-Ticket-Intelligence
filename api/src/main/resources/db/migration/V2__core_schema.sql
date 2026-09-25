-- Core schema. Design notes in docs/IMPLEMENTATION_PLAN.md section 6.

CREATE TABLE agents (
  id            BIGSERIAL PRIMARY KEY,
  username      TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  display_name  TEXT NOT NULL,
  role          TEXT NOT NULL CHECK (role IN ('AGENT','SENIOR','ADMIN')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE customers (
  id          BIGSERIAL PRIMARY KEY,
  external_id TEXT UNIQUE NOT NULL,
  name        TEXT,
  tier        TEXT NOT NULL CHECK (tier IN ('FREE','PRO','ENTERPRISE')),
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sla_policies (             -- first-response SLA minutes by tier x priority
  tier     TEXT NOT NULL,
  priority TEXT NOT NULL,
  minutes  INT  NOT NULL,
  PRIMARY KEY (tier, priority)
);
INSERT INTO sla_policies (tier, priority, minutes) VALUES
  ('FREE','LOW',2880),('FREE','MEDIUM',1440),('FREE','HIGH',720),('FREE','URGENT',480),
  ('PRO','LOW',1440),('PRO','MEDIUM',480),('PRO','HIGH',240),('PRO','URGENT',120),
  ('ENTERPRISE','LOW',480),('ENTERPRISE','MEDIUM',240),('ENTERPRISE','HIGH',60),('ENTERPRISE','URGENT',15);

CREATE TABLE ingest_batches (
  id         BIGSERIAL PRIMARY KEY,
  filename   TEXT,
  status     TEXT NOT NULL,             -- PROCESSING | DONE | FAILED
  total      INT NOT NULL DEFAULT 0,
  accepted   INT NOT NULL DEFAULT 0,
  duplicates INT NOT NULL DEFAULT 0,
  rejected   INT NOT NULL DEFAULT 0,
  errors     JSONB,
  created_by TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE duplicate_clusters (
  id                  BIGSERIAL PRIMARY KEY,
  product             TEXT NOT NULL,
  title               TEXT,
  size                INT NOT NULL DEFAULT 1,
  is_incident         BOOLEAN NOT NULL DEFAULT false,
  incident_flagged_at TIMESTAMPTZ,
  status              TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE','CLOSED','MERGED')),
  merged_into         BIGINT,
  first_seen_at       TIMESTAMPTZ NOT NULL,
  last_seen_at        TIMESTAMPTZ NOT NULL
);

CREATE TABLE tickets (
  id                BIGSERIAL PRIMARY KEY,
  external_id       TEXT UNIQUE,
  source            TEXT NOT NULL CHECK (source IN ('API','CSV','SEED')),
  ingest_batch_id   BIGINT REFERENCES ingest_batches(id),
  customer_id       BIGINT REFERENCES customers(id),
  customer_tier     TEXT NOT NULL CHECK (customer_tier IN ('FREE','PRO','ENTERPRISE')),
  product           TEXT NOT NULL,
  subject           TEXT NOT NULL,
  body              TEXT NOT NULL,
  content_hash      CHAR(64) NOT NULL,
  created_at        TIMESTAMPTZ NOT NULL,
  ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  triaged_at        TIMESTAMPTZ,
  drafted_at        TIMESTAMPTZ,
  status            TEXT NOT NULL DEFAULT 'NEW' CHECK (status IN
                    ('NEW','TRIAGING','TRIAGED','TRIAGE_FAILED','DRAFTED','IN_REVIEW','RESOLVED','CLOSED')),
  draft_state       TEXT NOT NULL DEFAULT 'NONE' CHECK (draft_state IN
                    ('NONE','PENDING','READY','SKIPPED','FAILED','WAITING_LLM')),
  category          TEXT,
  category_conf     REAL,
  category_source   TEXT CHECK (category_source IN ('MODEL','LLM','AGENT')),
  priority          TEXT,
  priority_conf     REAL,
  priority_source   TEXT CHECK (priority_source IN ('MODEL','LLM','AGENT')),
  escalation_risk   REAL,
  queue             TEXT NOT NULL DEFAULT 'STANDARD' CHECK (queue IN ('STANDARD','SENIOR')),
  cluster_id        BIGINT REFERENCES duplicate_clusters(id),
  sla_due_at        TIMESTAMPTZ,
  sla_breached      BOOLEAN NOT NULL DEFAULT false,
  first_response_at TIMESTAMPTZ,
  assigned_agent_id BIGINT REFERENCES agents(id),
  resolution        TEXT,
  resolved_at       TIMESTAMPTZ,
  version           INT NOT NULL DEFAULT 0,
  search_tsv        TSVECTOR GENERATED ALWAYS AS
                    (to_tsvector('english', subject || ' ' || body || ' ' || coalesce(resolution, ''))) STORED
);
CREATE INDEX tickets_queue_idx    ON tickets (status, queue, escalation_risk DESC NULLS LAST, id DESC);
CREATE INDEX tickets_product_ts   ON tickets (product, created_at DESC);
CREATE INDEX tickets_customer_ts  ON tickets (customer_id, created_at DESC);
CREATE INDEX tickets_cluster_idx  ON tickets (cluster_id) WHERE cluster_id IS NOT NULL;
CREATE INDEX tickets_tsv_idx      ON tickets USING gin (search_tsv);
CREATE INDEX tickets_hash_idx     ON tickets (content_hash);

ALTER TABLE duplicate_clusters ADD COLUMN canonical_ticket_id BIGINT REFERENCES tickets(id);

CREATE TABLE ticket_embeddings (       -- slim table keeps ticket rows small
  ticket_id     BIGINT PRIMARY KEY REFERENCES tickets(id) ON DELETE CASCADE,
  model_version TEXT NOT NULL,
  is_resolved   BOOLEAN NOT NULL DEFAULT false,
  is_open       BOOLEAN NOT NULL DEFAULT true,
  product       TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL,
  embedding     VECTOR(384) NOT NULL
);
-- Partial HNSW indexes: RAG corpus (resolved) and dedup candidates (open).
CREATE INDEX emb_resolved_hnsw ON ticket_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64) WHERE is_resolved;
CREATE INDEX emb_open_hnsw ON ticket_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64) WHERE is_open;
CREATE INDEX emb_open_time ON ticket_embeddings (product, created_at DESC) WHERE is_open;

CREATE TABLE predictions (             -- append-only history
  id            BIGSERIAL PRIMARY KEY,
  ticket_id     BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
  task          TEXT NOT NULL CHECK (task IN ('CATEGORY','PRIORITY','ESCALATION')),
  model_name    TEXT NOT NULL,
  model_version TEXT NOT NULL,
  label         TEXT,
  score         REAL NOT NULL,
  details       JSONB,
  latency_ms    INT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX predictions_idem ON predictions (ticket_id, task, model_name, model_version);
CREATE INDEX predictions_ticket_idx ON predictions (ticket_id, task, created_at DESC);

CREATE TABLE duplicate_links (
  ticket_id  BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
  similar_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
  similarity REAL NOT NULL,
  PRIMARY KEY (ticket_id, similar_id)
);

CREATE TABLE drafts (
  id                   BIGSERIAL PRIMARY KEY,
  ticket_id            BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
  status               TEXT NOT NULL CHECK (status IN ('GENERATED','FAILED','SKIPPED','APPROVED','EDITED','REJECTED','SUPERSEDED')),
  body                 TEXT,
  citations            JSONB,
  grounding            TEXT,
  llm_confidence       REAL,
  needs_info           JSONB,
  sources              JSONB,            -- retrieved tickets shown to the model (id, subject, score)
  reused_from_draft_id BIGINT REFERENCES drafts(id),
  instruction          TEXT,
  model                TEXT,
  prompt_version       TEXT,
  input_tokens         INT,
  output_tokens        INT,
  latency_ms           INT,
  cost_usd             NUMERIC(10,6),
  error                TEXT,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX drafts_ticket_idx ON drafts (ticket_id, created_at DESC);

CREATE TABLE feedback (
  id                 BIGSERIAL PRIMARY KEY,
  ticket_id          BIGINT NOT NULL REFERENCES tickets(id),
  draft_id           BIGINT REFERENCES drafts(id),
  agent_id           BIGINT REFERENCES agents(id),
  action             TEXT NOT NULL CHECK (action IN ('APPROVE','EDIT_APPROVE','REJECT','RELABEL')),
  final_body         TEXT,
  edit_distance      INT,
  edit_ratio         REAL,
  reject_reason      TEXT,
  corrected_category TEXT,
  corrected_priority TEXT,
  rating             SMALLINT CHECK (rating BETWEEN 1 AND 5),
  time_to_review_ms  BIGINT,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX feedback_ticket_idx ON feedback (ticket_id);

CREATE TABLE jobs (
  id           BIGSERIAL PRIMARY KEY,
  type         TEXT NOT NULL CHECK (type IN ('TRIAGE','LLM_CLASSIFY','DRAFT','REEMBED')),
  ticket_id    BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
  status       TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING','RUNNING','DONE','DEAD')),
  priority     INT  NOT NULL DEFAULT 0,
  attempts     INT  NOT NULL DEFAULT 0,
  max_attempts INT  NOT NULL DEFAULT 5,
  run_after    TIMESTAMPTZ NOT NULL DEFAULT now(),
  locked_by    TEXT,
  locked_at    TIMESTAMPTZ,
  last_error   TEXT,
  payload      JSONB,                    -- e.g. draft regeneration instruction
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX jobs_ready_idx ON jobs (type, priority DESC, id) WHERE status = 'PENDING';
CREATE INDEX jobs_running_idx ON jobs (locked_at) WHERE status = 'RUNNING';
CREATE UNIQUE INDEX jobs_one_active ON jobs (ticket_id, type) WHERE status IN ('PENDING','RUNNING');
