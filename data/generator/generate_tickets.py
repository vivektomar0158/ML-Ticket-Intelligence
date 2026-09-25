"""Step 3: turn specs into ticket text (subject, body, agent resolution) with Gemini.

Batches of 10 specs per call. Batches never cross a split boundary and incident tickets are batched
per incident, so paraphrases written in one call stay in the same split (no leakage).
Everything is disk-cached (llm.py), so re-running resumes for free.

  uv run --project ml-service python -m data.generator.generate_tickets --pilot 12
  uv run --project ml-service python -m data.generator.generate_tickets            # full run
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from pydantic import BaseModel

from data.generator.llm import ROOT, generate_json, usage_totals
from data.generator.sample_specs import load

BATCH = 10
OUT = ROOT / "data/raw/tickets_raw.parquet"


class Out(BaseModel):
    idx: int
    subject: str
    body: str
    resolution: str


TONE_HINT = {
    "neutral": "matter-of-fact", "polite": "courteous and appreciative", "frustrated": "visibly frustrated",
    "angry": "angry, may use caps or exclamation marks, but no profanity", "terse": "very short and curt",
    "rambling": "long-winded with irrelevant details before getting to the point",
}
QUALITY_HINT = {
    "clean": "clean grammar", "typos": "a few realistic typos and missing punctuation",
    "all_lowercase": "all lowercase, minimal punctuation, like typed on a phone",
    "non_native": "slightly non-native English phrasing",
}
PRIO_HINT = {
    "LOW": "no urgency; casual, can wait",
    "MEDIUM": "normal importance, wants it fixed soon but not pressing",
    "HIGH": "important, blocking their work; asks for prompt help",
    "URGENT": "critical: many users or revenue affected, or a hard immediate deadline; conveys strong urgency",
}
AMBIG_HINT = {
    "clear": "",
    "vague": " The ticket must be VAGUE: one or two short sentences, missing details (e.g. 'it's not working, please help').",
    "multi_issue": " The customer ALSO briefly mentions a second, unrelated minor problem, but the main issue is the one above.",
}


def spec_line(i, r, rc):
    parts = [
        f"#{i}. Issue: {rc['symptoms']}",
        f"Product: {r.product}. Customer plan: {r.customer_tier}. Writer: {r.persona}.",
        f"Tone: {TONE_HINT[r.tone]}. Writing: {QUALITY_HINT[r.writing_quality]}.",
        f"Urgency: {PRIO_HINT[r.priority]}.",
    ]
    if rc["error_code"]:
        parts.append(f"Error code seen: {rc['error_code']} (mention it in roughly half of cases).")
    if r.has_deadline:
        parts.append("Include a concrete deadline or time pressure (e.g. a demo at 5pm today, audit on Friday, payroll run tomorrow).")
    if r.incident_id:
        parts.append("Many customers are hitting this at the same moment right now; describe it in your own words, do not mention others.")
    parts.append(AMBIG_HINT[r.ambiguity].strip())
    return " ".join(p for p in parts if p)


def make_prompt(chunk: pd.DataFrame, rc_by_id: dict, product_desc: str) -> str:
    specs = "\n".join(spec_line(i, r, rc_by_id[r.root_cause_id]) for i, r in enumerate(chunk.itertuples()))
    fixes = "\n".join(f"#{i}: {rc_by_id[r.root_cause_id]['canonical_fix']}" for i, r in enumerate(chunk.itertuples()))
    return f"""You write realistic customer support tickets for a fictional SaaS product, plus the reply an agent sent.
{product_desc}

Write exactly {len(chunk)} tickets, one per spec below, in the same order. For each return:
- idx: the spec number
- subject: what a customer would type, max 12 words, varied style (not always a full sentence)
- body: the customer's message, 1-7 sentences depending on tone/quality. Do NOT include labels, category names,
  ticket ids or the words 'urgent priority'. Do not name the customer's plan unless natural. Vary openings/closings.
- resolution: the support agent's reply that resolved it (2-5 sentences), based ONLY on the canonical fix for that spec,
  reworded naturally (do not copy verbatim), addressing this customer's situation.

SPECS:
{specs}

CANONICAL FIXES (for resolution):
{fixes}
"""


def run_batch(args):
    gid, chunk, rc_by_id, product_desc = args
    prompt = make_prompt(chunk, rc_by_id, product_desc)
    for attempt in range(3):
        out = generate_json(prompt, schema=list[Out], temperature=0.95, tag=f"tix-{gid}-{attempt}")
        by_idx = {o["idx"]: o for o in out}
        if all(i in by_idx for i in range(len(chunk))):
            rows = []
            for i, r in enumerate(chunk.itertuples()):
                o = by_idx[i]
                rows.append({"ticket_uid": r.ticket_uid, "group_id": gid, "subject": o["subject"].strip(),
                             "body": o["body"].strip(), "resolution": o["resolution"].strip()})
            return rows
        print(f"  batch {gid}: missing idx, retry")
    return []


def make_batches(df: pd.DataFrame, seed: int) -> list[tuple[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    batches = []
    for inc, g in df[df.incident_id.notna()].groupby("incident_id"):
        # an incident is only ever in one split (split follows its start), so chunk purely by incident
        for k in range(0, len(g), BATCH):
            batches.append((f"{inc}-{k // BATCH}", g.iloc[k:k + BATCH]))
    for split, g in df[df.incident_id.isna()].groupby("split"):
        g = g.iloc[rng.permutation(len(g))]
        for k in range(0, len(g), BATCH):
            batches.append((f"{split}-{k // BATCH}", g.iloc[k:k + BATCH]))
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="only generate N random batches")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    tax, rcs = load()
    rc_by_id = {r["id"]: r for r in rcs}
    specs = pd.read_parquet(ROOT / "data/raw/specs.parquet")
    batches = make_batches(specs, 7)
    if a.pilot:
        rng = np.random.default_rng(1)
        batches = [batches[i] for i in rng.permutation(len(batches))[:a.pilot]]
    print(f"{len(batches)} batches, {sum(len(b) for _, b in batches)} tickets")

    jobs = [(gid, ch, rc_by_id, tax["product_description"]) for gid, ch in batches]
    rows, done = [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(run_batch, jobs):
            rows.extend(res)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)} batches")
    text = pd.DataFrame(rows)
    df = specs.merge(text, on="ticket_uid", how="inner")
    out = OUT if not a.pilot else ROOT / "data/raw/tickets_pilot.parquet"
    df.to_parquet(out)
    print(f"{len(df)} tickets -> {out}", usage_totals())


if __name__ == "__main__":
    main()
