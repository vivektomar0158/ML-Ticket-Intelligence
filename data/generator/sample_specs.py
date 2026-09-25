"""Step 2: sample structured ticket specs (all ground-truth labels) deterministically.

Text is written later by Gemini from these specs, so labels are known by construction.
Escalation / SLA outcomes come from a latent function + noise (see logit below).
"""
import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yaml

from data.generator.llm import ROOT

START = datetime(2026, 6, 1, tzinfo=timezone.utc)  # a Monday
DAYS = 90
TRAIN_END, VAL_END = 63, 72  # day indices: [0,63) train, [63,72) val, [72,90) test
HOUR_W = np.array([1, 1, 1, 1, 1, 2, 3, 6, 10, 12, 12, 11, 9, 11, 12, 12, 10, 8, 5, 3, 2, 2, 1, 1], float)
PRIO = ["LOW", "MEDIUM", "HIGH", "URGENT"]
NEG = {"angry": 1.0, "frustrated": 0.6, "terse": 0.2}


def load():
    tax = yaml.safe_load((ROOT / "data/generator/taxonomy.yaml").read_text(encoding="utf-8"))
    rcs = json.loads((ROOT / "data/raw/root_causes.json").read_text(encoding="utf-8"))
    return tax, rcs


def pick(rng, d: dict):
    keys = list(d)
    p = np.array([d[k] for k in keys], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def sample_time(rng) -> datetime:
    while True:
        day = int(rng.integers(0, DAYS))
        dt = START + timedelta(days=day)
        if rng.random() < (1.0 if dt.weekday() < 5 else 0.3):
            hour = int(rng.choice(24, p=HOUR_W / HOUR_W.sum()))
            return dt + timedelta(hours=hour, minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))


def split_of(ts: datetime) -> str:
    day = (ts - START).days
    return "train" if day < TRAIN_END else "val" if day < VAL_END else "test"


def build(n: int, seed: int, n_incidents: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tax, rcs = load()
    cats = tax["categories"]
    by_cat = {c: [r for r in rcs if r["category"] == c] for c in cats}
    # skewed popularity within a category: some issues are much more common than others
    rc_w = {c: (1 / np.arange(1, len(v) + 1) ** 0.6)[rng.permutation(len(v))] for c, v in by_cat.items()}

    # customers: heavy-tailed activity, enterprise customers file more
    n_cust = 500
    cust_tier = [pick(rng, tax["tiers"]) for _ in range(n_cust)]
    act = rng.pareto(1.5, n_cust) + 1
    act *= np.array([{"FREE": 1, "PRO": 1.5, "ENTERPRISE": 3}[t] for t in cust_tier])
    cust_p = act / act.sum()

    # ---- incidents: outage-like root causes, each emits a burst of paraphrased tickets
    cand = [r for r in rcs if r["category"] in ("BUG", "PERFORMANCE", "LOGIN_ACCESS", "INTEGRATION")
            and (r["typical_priority"] in ("HIGH", "URGENT") or r["is_release_related"])]
    inc_rcs = [cand[i] for i in rng.permutation(len(cand))[:n_incidents]]
    incidents = []
    for k, rc in enumerate(inc_rcs):
        while True:
            t0 = sample_time(rng)
            if t0.weekday() < 5 and 8 <= t0.hour <= 17 and t0 + timedelta(hours=3) < START + timedelta(days=DAYS):
                break
        incidents.append({"id": f"INC-{k + 1:02d}", "rc": rc, "t0": t0,
                          "size": int(rng.integers(20, 61)), "dur_min": int(rng.integers(30, 91))})
    windows = [(i["rc"]["id"], i["t0"] - timedelta(hours=6),
                i["t0"] + timedelta(minutes=i["dur_min"]) + timedelta(hours=6)) for i in incidents]

    rows = []

    def base_row(rc, ts, incident_id):
        prod = rc["product"] if rng.random() < 0.9 else str(rng.choice(tax["products"]))
        return {"category": rc["category"], "root_cause_id": rc["id"], "product": prod,
                "created_at": ts, "incident_id": incident_id}

    cat_keys = list(cats)
    cat_p = np.array([cats[c]["weight"] for c in cat_keys])
    cat_p /= cat_p.sum()
    n_inc = sum(i["size"] for i in incidents)
    n_bg = 0
    while n_bg < n - n_inc:
        c = cat_keys[rng.choice(len(cat_keys), p=cat_p)]
        w = rc_w[c] / rc_w[c].sum()
        rc = by_cat[c][rng.choice(len(w), p=w)]
        ts = sample_time(rng)
        if any(rid == rc["id"] and a <= ts <= b for rid, a, b in windows):
            continue  # keep incident ground truth clean: no unlabeled near-duplicates around a burst
        rows.append(base_row(rc, ts, None))
        n_bg += 1
    for inc in incidents:
        # ramp-up shaped arrival: more tickets early in the window
        offs = np.sort(inc["dur_min"] * rng.beta(1.3, 2.0, inc["size"]))
        for o in offs:
            rows.append(base_row(inc["rc"], inc["t0"] + timedelta(minutes=float(o)), inc["id"]))

    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    N = len(df)

    cidx = rng.choice(n_cust, N, p=cust_p)
    df["customer_id"] = [f"C-{i + 1:04d}" for i in cidx]
    df["customer_tier"] = [cust_tier[i] for i in cidx]
    df["persona"] = rng.choice(tax["personas"], N)
    df["tone"] = [pick(rng, tax["tones"]) for _ in range(N)]
    df["writing_quality"] = [pick(rng, tax["writing_quality"]) for _ in range(N)]
    dl_p = df.category.map({"BUG": 0.18, "LOGIN_ACCESS": 0.18, "INTEGRATION": 0.14, "PERFORMANCE": 0.12,
                            "DATA_PRIVACY": 0.10, "BILLING": 0.08, "ACCOUNT_MANAGEMENT": 0.08,
                            "FEATURE_REQUEST": 0.02})
    df["has_deadline"] = rng.random(N) < dl_p.values
    df["ambiguity"] = [("vague" if u < 0.06 else "multi_issue" if u < 0.11 else "clear") if inc is None else "clear"
                       for u, inc in zip(rng.random(N), df.incident_id)]

    # ---- priority: root-cause default, bumped by deadline / incident, plus label noise
    rc_prio = {r["id"]: PRIO.index(r["typical_priority"]) for r in rcs}
    p = df.root_cause_id.map(rc_prio).values.astype(int) - (rng.random(N) < 0.4)  # most issues are less severe than the worst case
    p = p + ((df.has_deadline.values) & (rng.random(N) < 0.7)) + ((df.incident_id.notna().values) & (rng.random(N) < 0.5))
    p = p + rng.choice([-1, 0, 1], N, p=[0.06, 0.88, 0.06])
    df["priority"] = [PRIO[int(np.clip(x, 0, 3))] for x in p]

    # ---- time features & escalation (latent function + noise)
    df["hour"] = df.created_at.map(lambda t: t.hour)
    df["weekday"] = df.created_at.map(lambda t: t.weekday())
    df["off_hours"] = (df.hour < 9) | (df.hour >= 18) | (df.weekday >= 5)
    secs = df.created_at.map(lambda t: int(t.timestamp())).values
    prior = np.zeros(N, int)
    for cid, idx in df.groupby("customer_id").indices.items():
        t = secs[idx]
        j = 0
        for k in range(len(idx)):
            while t[k] - t[j] > 7 * 86400:
                j += 1
            prior[idx[k]] = k - j
    df["prior_tickets_7d"] = prior

    prio_i = df.priority.map(PRIO.index).values
    logit = (-4.0
             + df.customer_tier.map({"FREE": 0, "PRO": 0.5, "ENTERPRISE": 1.2}).values
             + 0.9 * (prio_i >= 2) + 0.7 * (prio_i == 3)
             + 0.9 * df.tone.map(NEG).fillna(0).values
             + 0.8 * df.has_deadline.values
             + 0.9 * df.incident_id.notna().values
             + 0.4 * df.off_hours.values
             + 0.25 * np.minimum(prior, 5)
             + 0.5 * (df.category == "DATA_PRIVACY").values
             + rng.normal(0, 0.5, N))
    prob = 1 / (1 + np.exp(-logit))
    df["escalated"] = rng.random(N) < prob
    df["sla_breached"] = rng.random(N) < 1 / (1 + np.exp(-(logit - 0.4 + rng.normal(0, 0.5, N))))
    df["split"] = df.created_at.map(split_of)
    df["ticket_uid"] = [f"T{seed}-{i:05d}" for i in range(N)]
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = build(a.n, a.seed)
    out = ROOT / "data/raw/specs.parquet"
    df.to_parquet(out)
    print(f"{len(df)} specs -> {out}")
    print(df.split.value_counts().to_dict(), "| incident tickets:", int(df.incident_id.notna().sum()))
    print("category:", df.category.value_counts(normalize=True).round(3).to_dict())
    print("priority:", df.priority.value_counts(normalize=True).round(3).to_dict())
    print("escalated rate:", round(df.escalated.mean(), 3), "| by tier:",
          df.groupby("customer_tier").escalated.mean().round(3).to_dict())
    print("sla_breached rate:", round(df.sla_breached.mean(), 3))
