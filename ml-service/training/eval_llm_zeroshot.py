"""Gemini zero-shot classification on a sample (free-tier quota makes the full test set impractical).

Two regimes, same protocol as the small models:
  in_dist : stratified sample of the standard test split (issues seen in training)
  unseen  : sample of tickets whose root cause was held out from training (see eval_robustness)
  uv run python -m training.eval_llm_zeroshot --n 120
"""
import argparse
import asyncio
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from app.llm.gemini import LlmUnavailable, gateway
from app.llm.pii import redact
from app.llm.prompts import CATEGORIES, CLASSIFY_VERSION, ClassifyOut, classify_prompt
from training.common import METRICS, load_metrics, load_tickets, save_metrics


async def run(df: pd.DataFrame, conc: int = 4) -> pd.DataFrame:
    rows, sem = [], asyncio.Semaphore(conc)

    async def one(r):
        async with sem:
            text, _ = redact(r.text)
            t = time.perf_counter()
            try:
                d, m = await gateway.generate_json(classify_prompt(text), ClassifyOut, version=CLASSIFY_VERSION)
            except LlmUnavailable as e:
                print("  LLM unavailable:", str(e)[:100])
                return
            cat = d["category"] if d["category"] in CATEGORIES else "BUG"
            rows.append({"ticket_uid": r.ticket_uid, "y": r.category, "llm_pred": cat, "llm_priority": d["priority"],
                         "llm_conf": float(d["confidence"]), "y_priority": r.priority, "model": m["model"],
                         "latency_ms": m["latency_ms"],  # original call latency (stored in the cache)
                         "in_tokens": m["input_tokens"], "out_tokens": m["output_tokens"], "cost_usd": m["cost_usd"]})

    await asyncio.gather(*[one(r) for r in df.itertuples()])
    return pd.DataFrame(rows)


def summarize(d: pd.DataFrame) -> dict:
    return {"n": len(d), "macro_f1": float(f1_score(d.y, d.llm_pred, average="macro")),
            "accuracy": float(accuracy_score(d.y, d.llm_pred)),
            "priority_accuracy": float(accuracy_score(d.y_priority, d.llm_priority)),
            "latency_p50_ms": float(d.latency_ms.dropna().quantile(.5)) if d.latency_ms.notna().any() else None,
            "latency_p95_ms": float(d.latency_ms.dropna().quantile(.95)) if d.latency_ms.notna().any() else None,
            "cost_per_1k_tickets_usd": float(d.cost_usd.mean() * 1000),
            "avg_in_tokens": float(d.in_tokens.mean()), "avg_out_tokens": float(d.out_tokens.mean()),
            "models_used": d.model.value_counts().to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    df, _ = load_tickets()
    held = load_metrics("robustness")["held_out_root_causes"]
    test = df[df.split == "test"]
    # stratified by category so rare classes are represented
    def strat(d):
        k = a.n // 8
        return pd.concat([g.sample(min(len(g), k), random_state=0) for _, g in d.groupby("category")])

    ind, uns = strat(test), strat(df[df.root_cause_id.isin(held)])
    # interleave the two regimes so a partially exhausted quota still yields both
    uns = uns[~uns.ticket_uid.isin(ind.ticket_uid)]  # a test ticket may also have a held-out root cause
    ind, uns = ind.assign(regime="in_dist"), uns.assign(regime="unseen")
    both = pd.concat([ind, uns]).sample(frac=1, random_state=1)
    d = asyncio.run(run(both))
    out = {}
    if len(d):
        reg = both.set_index("ticket_uid").regime
        d["regime"] = d.ticket_uid.map(reg)
        for name in ["in_dist", "unseen"]:
            x = d[d.regime == name]
            if len(x):
                x.to_parquet(METRICS / f"llm_zeroshot_{name}.parquet")
                out[name] = summarize(x)
                out[name]["planned_n"] = int((both.regime == name).sum())
                print(name, {k: v for k, v in out[name].items() if k != "models_used"})
    print("gateway stats:", gateway.stats)
    if out:
        save_metrics("llm_zeroshot", out)


if __name__ == "__main__":
    main()
