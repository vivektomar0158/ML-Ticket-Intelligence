"""Load test: steady traffic, then a 500-ticket outage-style burst. Measures ingest latency and time-to-triage / time-to-draft.

  scripts/reset-live.sh && uv run --project ml-service python infra/loadtest.py
Targets (plan section 1): ingest p95 < 50 ms; triage p95 < 2 s at normal load; draft p95 < 15 s; 500-ticket burst triaged < 60 s.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PG = "ticketintelligence-postgres-1"


def sql(q: str) -> str:
    return subprocess.run(["docker", "exec", PG, "psql", "-U", "ticketintel", "-d", "ticketintel", "-tAc", q], capture_output=True, text=True).stdout.strip()


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else None


def wait_drained(timeout=300) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if sql("select count(*) from jobs where status in ('PENDING','RUNNING')") == "0":
            return time.time() - t0
        time.sleep(0.5)
    return float("inf")


def lags(since_marker: str) -> dict:
    """Per-ticket pipeline lags in seconds from the DB timestamps (ingest -> triaged, ingest -> drafted)."""
    rows = sql(f"""select extract(epoch from triaged_at - ingested_at), coalesce(extract(epoch from drafted_at - ingested_at), -1)
                   from tickets where source <> 'SEED' and external_id like '{since_marker}%'""").splitlines()
    tri, dra = [], []
    for r in rows:
        a, b = r.split("|")
        tri.append(float(a))
        if float(b) >= 0:
            dra.append(float(b))
    return {"n": len(tri), "triage_p50_s": pct(tri, 50), "triage_p95_s": pct(tri, 95), "triage_max_s": max(tri) if tri else None,
            "draft_p50_s": pct(dra, 50), "draft_p95_s": pct(dra, 95), "drafted": len(dra)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--steady-rate", type=float, default=2.0)
    ap.add_argument("--steady-seconds", type=int, default=45)
    ap.add_argument("--burst", type=int, default=500)
    a = ap.parse_args()

    df = pd.read_parquet(ROOT / "data/processed/test.parquet")
    df = df[df.incident_id.isna()].sample(frac=1, random_state=1).reset_index(drop=True)
    c = httpx.Client(base_url=a.api, timeout=60)
    c.headers["Authorization"] = "Bearer " + c.post("/api/auth/login", json={"username": "agent", "password": "agent"}).json()["accessToken"]

    def payload(r, tag, i):
        return {"externalId": f"{tag}-{i}", "subject": r.subject, "body": r.body, "product": r.product, "customerId": r.customer_id, "customerTier": r.customer_tier}

    out = {}
    # ---------------------------------------------------------------- steady load
    n = int(a.steady_rate * a.steady_seconds)
    lat = []
    t0 = time.time()
    for i, r in enumerate(df.head(n).itertuples()):
        target = t0 + i / a.steady_rate
        time.sleep(max(0, target - time.time()))
        s = time.perf_counter()
        c.post("/api/tickets", json=payload(r, "steady", i)).raise_for_status()
        lat.append((time.perf_counter() - s) * 1000)
    drain = wait_drained()
    out["steady"] = {"rate_per_s": a.steady_rate, "tickets": n, "ingest_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99)},
                     "drain_after_last_s": drain, **lags("steady-")}
    print("steady:", json.dumps(out["steady"], indent=1))

    # ---------------------------------------------------------------- burst (outage-style)
    lat, sent = [], 0
    t0 = time.time()
    rows = df.iloc[n:n + a.burst]
    for i in range(0, len(rows), 50):
        chunk = rows.iloc[i:i + 50]
        s = time.perf_counter()
        c.post("/api/tickets/batch", json=[payload(r, "burst", i + j) for j, r in enumerate(chunk.itertuples())]).raise_for_status()
        lat.append((time.perf_counter() - s) * 1000)
        sent += len(chunk)
    ingest_s = time.time() - t0
    t_triaged = None
    while time.time() - t0 < 300:
        left = int(sql("select count(*) from tickets where external_id like 'burst-%' and triaged_at is null") or 1)
        if left == 0:
            t_triaged = time.time() - t0
            break
        time.sleep(0.25)
    drain = wait_drained()
    out["burst"] = {"tickets": sent, "ingest_seconds": ingest_s, "ingest_throughput_per_s": sent / ingest_s, "batch_of_50_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95)},
                    "all_triaged_after_s": t_triaged, "pipeline_drained_after_s": time.time() - t0, **lags("burst-")}
    states = sql("select draft_state || ':' || count(*) from tickets where external_id like 'burst-%' group by draft_state")
    out["burst"]["draft_states"] = states.replace("\n", ", ")
    print("burst:", json.dumps(out["burst"], indent=1))

    (ROOT / "eval/metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "eval/metrics/loadtest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
