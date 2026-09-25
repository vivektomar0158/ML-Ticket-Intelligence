"""Seed + replay tool for demos and load tests.

  # 1) load the resolved history (RAG corpus) with precomputed embeddings
  uv run --project ml-service python scripts/seed.py history
  # 2) replay unseen (test-split) tickets as a live stream; times shifted so the stream ends "now"
  uv run --project ml-service python scripts/seed.py replay --n 300 --speed 0        # as fast as possible
  uv run --project ml-service python scripts/seed.py replay --n 300 --speed 60       # 60x real time
  # 3) replay one simulated outage (a burst of similar tickets) - watch the incident appear
  uv run --project ml-service python scripts/seed.py incident --speed 30
"""
import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def client(api: str) -> httpx.Client:
    c = httpx.Client(base_url=api, timeout=120)
    r = c.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    r.raise_for_status()
    c.headers["Authorization"] = f"Bearer {r.json()['accessToken']}"
    return c


def iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def cmd_history(a):
    c = client(a.api)
    df = pd.read_parquet(DATA / "tickets.parquet").reset_index(drop=True)
    E = np.load(DATA / "emb_minilm.npy")
    hist = df[df.is_history]
    if a.limit:
        hist = hist.head(a.limit)
    imported = skipped = 0
    for i in range(0, len(hist), 200):
        chunk = hist.iloc[i:i + 200]
        items = [{"externalId": r.ticket_uid, "createdAt": iso(r.created_at), "customerId": r.customer_id,
                  "customerTier": r.customer_tier, "product": r.product, "subject": r.subject, "body": r.body,
                  "resolution": r.resolution, "category": r.category, "priority": r.priority,
                  "embedding": [float(x) for x in E[idx]]} for idx, r in zip(chunk.index, chunk.itertuples())]
        res = c.post("/api/admin/history", json=items)
        res.raise_for_status()
        imported += res.json()["imported"]
        skipped += res.json()["skipped"]
        print(f"  {i + len(chunk)}/{len(hist)} imported={imported} skipped={skipped}", end="\r")
    print(f"\nhistory loaded: {imported} imported, {skipped} skipped")


def replay(c: httpx.Client, rows: pd.DataFrame, speed: float, batch: int):
    """Send tickets in chronological order; timestamps shifted so the last ticket is 'now'; speed=0 -> no waiting."""
    rows = rows.sort_values("created_at").reset_index(drop=True)
    shift = datetime.now(timezone.utc) - pd.Timestamp(rows.created_at.iloc[-1]).to_pydatetime()
    sent, t_prev = 0, None
    for i in range(0, len(rows), batch):
        chunk = rows.iloc[i:i + batch]
        if speed > 0 and t_prev is not None:
            gap = (pd.Timestamp(chunk.created_at.iloc[0]) - t_prev).total_seconds() / speed
            time.sleep(min(gap, 30))
        t_prev = pd.Timestamp(chunk.created_at.iloc[-1])
        payload = [{"externalId": f"replay-{time.time_ns()}-{r.ticket_uid}", "subject": r.subject, "body": r.body, "product": r.product,
                    "customerId": r.customer_id, "customerTier": r.customer_tier,
                    "createdAt": iso(pd.Timestamp(r.created_at) + shift)} for r in chunk.itertuples()]
        res = c.post("/api/tickets/batch", json=payload)
        res.raise_for_status()
        sent += res.json()["created"]
        print(f"  sent {sent}/{len(rows)}", end="\r")
    print(f"\nreplayed {sent} tickets")


def cmd_replay(a):
    c = client(a.api)
    df = pd.read_parquet(DATA / "test.parquet")
    df = df[df.incident_id.isna()] if a.no_incidents else df
    df = df.sort_values("created_at").head(a.n)
    replay(c, df, a.speed, a.batch)


def cmd_incident(a):
    c = client(a.api)
    df = pd.read_parquet(DATA / "test.parquet")
    inc = df[df.incident_id.notna()]
    pick = a.id or inc.incident_id.value_counts().index[0]
    rows = inc[inc.incident_id == pick]
    print(f"replaying incident {pick}: {len(rows)} tickets over {(rows.created_at.max() - rows.created_at.min())}")
    replay(c, rows, a.speed, a.batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("history"); h.add_argument("--limit", type=int, default=0); h.set_defaults(fn=cmd_history)
    r = sub.add_parser("replay"); r.add_argument("--n", type=int, default=200); r.add_argument("--speed", type=float, default=0)
    r.add_argument("--batch", type=int, default=20); r.add_argument("--no-incidents", action="store_true"); r.set_defaults(fn=cmd_replay)
    i = sub.add_parser("incident"); i.add_argument("--id"); i.add_argument("--speed", type=float, default=0)
    i.add_argument("--batch", type=int, default=5); i.set_defaults(fn=cmd_incident)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
