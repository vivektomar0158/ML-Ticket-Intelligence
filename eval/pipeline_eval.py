"""End-to-end evaluation of the RUNNING system against ground truth.

Prerequisite: history loaded (scripts/seed.py history) and the test split replayed (scripts/seed.py replay --n 5000 --batch 50),
pipeline drained. Everything below is read back through the public API, i.e. it scores what an agent would actually see.

  uv run --project ml-service python eval/pipeline_eval.py
"""
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml-service"))
from training.common import save_metrics  # noqa: E402

API = "http://localhost:8080"


def client() -> httpx.Client:
    c = httpx.Client(base_url=API, timeout=120)
    tok = c.post("/api/auth/login", json={"username": "admin", "password": "admin"}).raise_for_status().json()["accessToken"]
    c.headers["Authorization"] = f"Bearer {tok}"
    return c


def all_tickets(c: httpx.Client, **params) -> list[dict]:
    out, cursor = [], None
    while True:
        r = c.get("/api/tickets", params={**params, "limit": 100, "sort": "newest", **({"cursor": cursor} if cursor else {})}).raise_for_status().json()
        out += r["items"]
        cursor = r["nextCursor"]
        if not cursor:
            return out


def main():
    truth = pd.read_parquet(ROOT / "data/processed/tickets.parquet").set_index("ticket_uid")
    c = client()

    live = [t for t in all_tickets(c, open=False) + all_tickets(c, open=True) if (t["external_id"] or "").startswith("replay-")]
    live = list({t["id"]: t for t in live}.values())
    seed_by_id = {t["id"]: t["external_id"] for t in all_tickets(c, open=False) if not (t["external_id"] or "").startswith("replay-")}
    df = pd.DataFrame(live)
    df["uid"] = df.external_id.str.split("-", n=2).str[-1]
    df = df[df.uid.isin(truth.index)].copy()
    for col in ["category", "priority", "root_cause_id", "incident_id", "escalated"]:
        df[f"true_{col}"] = df.uid.map(truth[col])
    print(f"scoring {len(df)} replayed tickets")
    out = {"n_tickets": len(df)}

    # ---------- details (model's own category prediction, draft sources, citations)
    model_cat, sources, cited = {}, {}, {}
    for tid in df.id:
        d = c.get(f"/api/tickets/{tid}").raise_for_status().json()
        cats = [p for p in d["predictions"] if p["task"] == "CATEGORY" and p["model_name"] != "gemini_zeroshot"]
        model_cat[tid] = cats[0]["label"] if cats else None
        dr = d.get("draft")
        if dr and dr.get("sources"):
            sources[tid] = [s["id"] for s in dr["sources"]]
            cited[tid] = [x["ticketId"] for x in (dr.get("citations") or [])]
    df["model_category"] = df.id.map(model_cat)

    # ---------- classification (model output, before any cascade override) + priority
    m = df.dropna(subset=["model_category"])
    out["category_model"] = {"macro_f1": float(f1_score(m.true_category, m.model_category, average="macro")),
                             "accuracy": float(accuracy_score(m.true_category, m.model_category)), "n": len(m)}
    out["category_served"] = {"accuracy": float(accuracy_score(df.true_category, df.category)),
                              "share_llm_classified": float((df.category_source == "LLM").mean())}
    out["priority"] = {"macro_f1": float(f1_score(df.true_priority, df.priority, average="macro")),
                       "accuracy": float(accuracy_score(df.true_priority, df.priority)),
                       "urgent_recall": float(((df.priority == "URGENT") & (df.true_priority == "URGENT")).sum() / max((df.true_priority == "URGENT").sum(), 1))}

    # ---------- escalation risk + routing
    e = df.dropna(subset=["escalation_risk"])
    senior = e.queue == "SENIOR"
    out["escalation"] = {"roc_auc": float(roc_auc_score(e.true_escalated.astype(int), e.escalation_risk)),
                         "senior_share": float(senior.mean()),
                         "senior_precision": float(e.true_escalated[senior].mean()) if senior.any() else None,
                         "senior_recall": float(e.true_escalated[senior].sum() / max(e.true_escalated.sum(), 1)),
                         "base_rate": float(e.true_escalated.mean())}

    # ---------- incidents / duplicate clusters
    inc_rows = df[df.true_incident_id.notna()]
    det = []
    for inc, g in inc_rows.groupby("true_incident_id"):
        counts = g.cluster_id.value_counts()
        top = counts.index[0] if len(counts) else None
        members = df[df.cluster_id == top] if top is not None else df.iloc[0:0]
        det.append({"incident": inc, "tickets": len(g), "in_top_cluster": int(counts.iloc[0]) if len(counts) else 0,
                    "recall": float(counts.iloc[0] / len(g)) if len(counts) else 0.0,
                    "purity": float((members.true_incident_id == inc).mean()) if len(members) else 0.0,
                    "flagged_incident": bool(g[g.cluster_id == top].is_incident.any()) if top is not None else False})
    flagged_clusters = df[df.is_incident].cluster_id.dropna().unique()
    false_flags = 0
    for cl in flagged_clusters:
        mem = df[df.cluster_id == cl]
        vc = mem.true_incident_id.fillna("none").value_counts()
        if vc.index[0] == "none" or vc.iloc[0] / len(mem) < 0.5:
            false_flags += 1
    out["incidents"] = {"per_incident": det, "detected": int(sum(d["flagged_incident"] for d in det)), "total": len(det),
                        "mean_member_recall": float(np.mean([d["recall"] for d in det])), "mean_purity": float(np.mean([d["purity"] for d in det])),
                        "flagged_clusters": int(len(flagged_clusters)), "false_incident_flags": int(false_flags)}

    # ---------- retrieval + grounding through the real Java hybrid retriever
    hist = truth.copy()
    rc_of_seed = {sid: hist.root_cause_id.get(uid) for sid, uid in seed_by_id.items()}
    hit1 = hit5 = mrr = n = 0
    cit_ok = cit_all = 0
    for tid, srcs in sources.items():
        true_rc = df.set_index("id").true_root_cause_id[tid]
        ranks = [rc_of_seed.get(s) == true_rc for s in srcs]
        n += 1
        hit1 += ranks[0] if ranks else 0
        hit5 += any(ranks[:5])
        first = next((i for i, x in enumerate(ranks) if x), None)
        mrr += 0 if first is None else 1 / (first + 1)
        for cid in cited.get(tid, []):
            cit_all += 1
            cit_ok += rc_of_seed.get(cid) == true_rc
    out["retrieval_e2e"] = {"drafted_tickets": n, "hit@1": hit1 / n, "hit@5": hit5 / n, "mrr": mrr / n,
                            "citation_precision": cit_ok / cit_all if cit_all else None, "citations": cit_all}

    # ---------- pipeline behaviour under the burst
    ov = c.get("/api/metrics/overview", params={"days": 1}).raise_for_status().json()
    states = df.groupby(["status", "draft_state"]).size().reset_index(name="n").to_dict("records")
    out["burst"] = {"final_states": states, "drafts_reused": ov["llm"]["reused_drafts"], "llm_draft_calls": ov["llm"]["draft_calls"],
                    "triage_latency_s": ov["latencySeconds"], "draft_latency_s": ov["draftLatencySeconds"]}
    save_metrics("pipeline", out)
    print(json.dumps({k: v for k, v in out.items() if k != "incidents"}, indent=1, default=str))
    print("incidents:", json.dumps({k: v for k, v in out["incidents"].items() if k != "per_incident"}, default=str))
    for d in out["incidents"]["per_incident"]:
        print("  ", d)


if __name__ == "__main__":
    main()
