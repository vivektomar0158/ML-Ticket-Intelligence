"""Cascade: cheap embedding classifier first, Gemini only when confidence < tau. Evaluated on the Gemini sample."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from training.common import METRICS, load_metrics, save_metrics


def cascade_table(d: pd.DataFrame, small_lat_ms: float, small: str = "emb"):
    rows = []
    llm_cost = d.cost_usd.mean()
    llm_lat = d.latency_ms.mean()
    for tau in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]:
        use_llm = d[f"{small}_conf"] < tau
        pred = np.where(use_llm, d.llm_pred, d[f"{small}_pred"])
        share = float(use_llm.mean())
        rows.append({"tau": tau, "llm_share": share, "macro_f1": float(f1_score(d.y, pred, average="macro")),
                     "accuracy": float(accuracy_score(d.y, pred)),
                     "cost_per_1k_usd": share * llm_cost * 1000,
                     "mean_latency_ms": (1 - share) * small_lat_ms + share * (small_lat_ms + llm_lat)})
    return rows


def main():
    cat = load_metrics("category")
    small_lat = cat["emb_lr"]["latency_b1"]["p50_ms"]
    out = {}
    src = {"in_dist": METRICS / "category_preds.parquet", "unseen": METRICS / "robust_preds.parquet"}
    for regime, path in src.items():
        f = METRICS / f"llm_zeroshot_{regime}.parquet"
        if not f.exists():
            out[regime] = {"status": "pending: no Gemini results (free-tier quota); rerun eval_llm_zeroshot after reset"}
            continue
        llm = pd.read_parquet(f)
        small = pd.read_parquet(path).drop(columns=[c for c in ["split", "y"] if c in pd.read_parquet(path).columns])
        d = llm.merge(small, on="ticket_uid")
        e = {"n": len(d), "small_only_accuracy": float(accuracy_score(d.y, d.emb_pred)),
             "small_only_macro_f1": float(f1_score(d.y, d.emb_pred, average="macro")),
             "llm_only_accuracy": float(accuracy_score(d.y, d.llm_pred)),
             "llm_only_macro_f1": float(f1_score(d.y, d.llm_pred, average="macro")),
             "table": cascade_table(d, small_lat)}
        lo, hi = d[d.emb_conf < 0.6], d[d.emb_conf >= 0.6]
        e["confidence_informative"] = {
            "acc_when_conf_lt_0.6": float((lo.emb_pred == lo.y).mean()) if len(lo) else None, "n_lt": len(lo),
            "acc_when_conf_ge_0.6": float((hi.emb_pred == hi.y).mean()) if len(hi) else None, "n_ge": len(hi)}
        out[regime] = e
        print(regime, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k != "table"})
    save_metrics("cascade", out)


if __name__ == "__main__":
    main()
