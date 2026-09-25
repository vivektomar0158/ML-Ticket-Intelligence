"""Duplicate-threshold sweep + retrieval ablation (vector vs keyword vs hybrid RRF).

Dedup ground truth: two tickets are duplicates if they belong to the same incident. We also report a looser
"same root cause within the window" view. Candidate pairs must be same product and within 72h (production rule).
Retrieval: queries = test tickets, corpus = train+val tickets (the resolved history); relevant = same root cause.
"""
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import re

from training.common import epoch_s, load_tickets, save_metrics

WINDOW_S = 72 * 3600


def pair_sweep(df, E, thresholds):
    ts = epoch_s(df.created_at)
    prod, inc, rc = df["product"].values, df.incident_id.values, df.root_cause_id.values
    order = np.argsort(ts)
    pi, pj, sims = [], [], []
    for pos, i in enumerate(order):
        lo = np.searchsorted(ts[order], ts[i] - WINDOW_S)
        cand = order[lo:pos]
        cand = cand[prod[cand] == prod[i]]
        if len(cand):
            s = E[cand] @ E[i]
            keep = s >= min(thresholds)
            pi += [i] * int(keep.sum()); pj += list(cand[keep]); sims += list(s[keep])
    pi, pj, sims = np.array(pi), np.array(pj), np.array(sims)
    same_inc = np.array([(inc[a] is not None and pd.notna(inc[a]) and inc[a] == inc[b]) for a, b in zip(pi, pj)])
    same_rc = rc[pi] == rc[pj]
    # positives that exist at all (same product + window), regardless of threshold: all such pairs
    tot_inc = same_inc.sum()
    rows = []
    for th in thresholds:
        m = sims >= th
        tp = (same_inc & m).sum()
        p, r = tp / max(m.sum(), 1), tp / max(tot_inc, 1)
        rc_p = (same_rc & m).sum() / max(m.sum(), 1)
        rows.append({"threshold": float(th), "incident_precision": float(p), "incident_recall": float(r),
                     "incident_f1": float(2 * p * r / max(p + r, 1e-9)), "same_root_cause_precision": float(rc_p),
                     "pairs": int(m.sum())})
    return rows


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def rrf(ranks_list, k=60, top=10):
    score = {}
    for ranks in ranks_list:
        for r, d in enumerate(ranks):
            score[d] = score.get(d, 0) + 1 / (k + r + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])[:top]]


def main():
    df, E = load_tickets()
    out = {}
    # ---------------- dedup threshold sweep on val (choose) and test (report)
    ths = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    val = df.split == "val"
    te = df.split == "test"
    sv = pair_sweep(df[val].reset_index(drop=True), E[val.values], ths)
    st = pair_sweep(df[te].reset_index(drop=True), E[te.values], ths)
    best = max(sv, key=lambda r: r["incident_f1"])
    out["dedup"] = {"val": sv, "test": st, "chosen_threshold": best["threshold"],
                    "test_at_chosen": next(r for r in st if r["threshold"] == best["threshold"])}
    print("dedup chosen tau:", best["threshold"], "| test:", {k: round(v, 3) for k, v in out["dedup"]["test_at_chosen"].items()})

    # ---------------- retrieval
    hist = df.split.isin(["train", "val"]).values
    q = df[df.split == "test"].reset_index(drop=True)
    Eh, Eq = E[hist], E[(df.split == "test").values]
    hdf = df[hist].reset_index(drop=True)
    hrc = hdf.root_cause_id.values
    corpus_docs = (hdf.subject + " " + hdf.body + " " + hdf.resolution).tolist()
    bm = BM25Okapi([tok(d) for d in corpus_docs])
    has_rel = np.isin(q.root_cause_id.values, hrc)
    print(f"queries {len(q)}, with a relevant doc in corpus {has_rel.mean():.3f}")

    ks = [1, 3, 5, 10]
    res = {m: {f"recall@{k}": [] for k in ks} | {"mrr": []} for m in ["vector", "keyword_bm25", "hybrid_rrf"]}
    for i in np.where(has_rel)[0]:
        rel = q.root_cause_id.iat[i]
        v = list(np.argsort(-(Eh @ Eq[i]))[:50])
        k = list(np.argsort(-bm.get_scores(tok(q.subject.iat[i] + " " + q.body.iat[i])))[:50])
        h = rrf([v[:20], k[:20]], top=50)
        for name, ranked in [("vector", v), ("keyword_bm25", k), ("hybrid_rrf", h)]:
            hit = [hrc[d] == rel for d in ranked]
            for kk in ks:
                res[name][f"recall@{kk}"].append(any(hit[:kk]))
            first = next((r for r, x in enumerate(hit) if x), None)
            res[name]["mrr"].append(1 / (first + 1) if first is not None else 0.0)
    out["retrieval"] = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()}
    out["retrieval"]["n_queries"] = int(has_rel.sum())
    out["retrieval"]["note"] = "relevant = same root cause; keyword leg is BM25 (proxy for Postgres full-text)."
    for m, d in out["retrieval"].items():
        if isinstance(d, dict):
            print(f"{m:13}", {k: round(v, 3) for k, v in d.items()})
    save_metrics("dedup_retrieval", out)


if __name__ == "__main__":
    main()
