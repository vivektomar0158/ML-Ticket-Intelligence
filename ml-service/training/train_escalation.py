"""Escalation-risk model: LightGBM on text signals + embedding PCA + predicted category/priority + metadata,
with Platt calibration fitted on the validation split.

  uv run python -m training.train_escalation
Serving-time feature parity: every feature here is computable at ingest (no post-handling information).
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from app.features.text import FEATURE_NAMES, extract_matrix
from training.common import (epoch_s, ART, CATEGORIES, PRIORITIES, ece, latency_ms, load_tickets, new_version, save_metrics)
from training.train_priority import build, tfidf_pipeline

PRODUCTS = ["web-app", "mobile-app", "api", "desktop-sync", "admin-console"]
TIER_ORD = {"FREE": 0, "PRO": 1, "ENTERPRISE": 2}
DEDUP_TAU, DEDUP_WINDOW_H = 0.70, 3


def similar_recent_counts(df: pd.DataFrame, E: np.ndarray) -> np.ndarray:
    """Cluster-size proxy computable at ingest: earlier tickets, same product, last 3h, cosine >= tau."""
    order = np.argsort(df.created_at.values)
    ts = epoch_s(df.created_at)
    out = np.zeros(len(df), np.float32)
    prod = df["product"].values
    for pos, i in enumerate(order):
        lo = np.searchsorted(ts[order], ts[i] - DEDUP_WINDOW_H * 3600)
        cand = order[lo:pos]
        cand = cand[prod[cand] == prod[i]]
        if len(cand):
            out[i] = float((E[cand] @ E[i] >= DEDUP_TAU).sum())
    return out


def meta_features(df: pd.DataFrame, cluster: np.ndarray) -> np.ndarray:
    return np.column_stack([
        df.customer_tier.map(TIER_ORD).values, df["product"].map(PRODUCTS.index).values, df.hour.values,
        df.weekday.values, df.off_hours.astype(int).values, df.prior_tickets_7d.values, cluster]).astype(np.float32)


META_NAMES = ["tier", "product", "hour", "weekday", "off_hours", "prior_tickets_7d", "similar_recent"]


def main():
    df, E = load_tickets()
    tr, va, te = (df.split == "train").values, (df.split == "val").values, (df.split == "test").values
    y = df.escalated.astype(int).values

    # --- predicted category / priority probabilities (out-of-fold on train so the GBM sees realistic, noisy inputs)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    cat_m = LogisticRegression(C=30, max_iter=3000, class_weight="balanced")
    P_cat = np.zeros((len(df), len(CATEGORIES)), np.float32)
    P_cat[tr] = cross_val_predict(cat_m, E[tr], df.category[tr], cv=skf, method="predict_proba")
    cat_m.fit(E[tr], df.category[tr])
    P_cat[~tr] = cat_m.predict_proba(E[~tr])
    assert list(cat_m.classes_) == sorted(CATEGORIES)

    feat = tfidf_pipeline(1).named_steps["f"]
    scaler = StandardScaler()
    yp = df.priority.map(PRIORITIES.index).values
    Xp_tr = build(feat, scaler, df.text[tr], df.customer_tier[tr], fit=True)
    Xp = {"va": build(feat, scaler, df.text[va], df.customer_tier[va]), "te": build(feat, scaler, df.text[te], df.customer_tier[te])}
    pri_m = LogisticRegression(C=1, max_iter=3000, class_weight="balanced")
    P_pri = np.zeros((len(df), 4), np.float32)
    P_pri[tr] = cross_val_predict(pri_m, Xp_tr, yp[tr], cv=skf, method="predict_proba")
    pri_m.fit(Xp_tr, yp[tr])
    P_pri[va], P_pri[te] = pri_m.predict_proba(Xp["va"]), pri_m.predict_proba(Xp["te"])

    pca = PCA(32, random_state=0).fit(E[tr])
    cluster = similar_recent_counts(df, E)
    S = extract_matrix(df.text.tolist())
    X = np.hstack([S, pca.transform(E), P_cat, P_pri, meta_features(df, cluster)]).astype(np.float32)
    names = (FEATURE_NAMES + [f"emb_pc{i}" for i in range(32)] + [f"p_cat_{c}" for c in sorted(CATEGORIES)]
             + [f"p_pri_{p}" for p in PRIORITIES] + META_NAMES)
    print("features:", X.shape, "| positives train/val/test:", y[tr].mean().round(3), y[va].mean().round(3), y[te].mean().round(3))

    # --- baseline: logistic regression on the same features
    sc = StandardScaler().fit(X[tr])
    base = LogisticRegression(C=1, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
    out = {"n_features": X.shape[1], "positive_rate": {"train": float(y[tr].mean()), "test": float(y[te].mean())},
           "logreg_baseline_auc": float(roc_auc_score(y[te], base.predict_proba(sc.transform(X[te]))[:, 1]))}

    # --- LightGBM: small random search on val AUC, early stopping
    rng = np.random.default_rng(0)
    best = None
    for trial in range(20):
        p = dict(objective="binary", learning_rate=0.03, num_leaves=int(rng.choice([7, 15, 31])),
                 min_child_samples=int(rng.choice([20, 40, 80])), subsample=float(rng.choice([0.7, 0.9])), subsample_freq=1,
                 colsample_bytree=float(rng.choice([0.5, 0.7, 0.9])), reg_lambda=float(rng.choice([0, 1, 5, 20])),
                 n_estimators=2000, random_state=trial, verbose=-1)
        m = lgb.LGBMClassifier(**p).fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
                                        callbacks=[lgb.early_stopping(80, verbose=False)])
        auc = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, p, m)
    val_auc, params, gbm = best
    print(f"best val AUC {val_auc:.4f}, iters {gbm.best_iteration_}, leaves {params['num_leaves']}")

    # --- calibration on val (Platt), isotonic reported for comparison
    raw_va, raw_te = gbm.predict_proba(X[va])[:, 1], gbm.predict_proba(X[te])[:, 1]
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1e6).fit(logit(raw_va).reshape(-1, 1), y[va])
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_va, y[va])
    cal_te = platt.predict_proba(logit(raw_te).reshape(-1, 1))[:, 1]
    iso_te = iso.predict(raw_te)

    def rel(p):  # reliability-curve points (10 equal-width bins)
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        return [{"bin": b, "mean_pred": float(p[idx == b].mean()), "frac_pos": float(y[te][idx == b].mean()),
                 "n": int((idx == b).sum())} for b in range(10) if (idx == b).any()]

    out.update({
        "val_auc": val_auc, "params": {k: v for k, v in params.items() if k != "verbose"}, "best_iteration": int(gbm.best_iteration_),
        "test": {
            "roc_auc": float(roc_auc_score(y[te], raw_te)), "pr_auc": float(average_precision_score(y[te], raw_te)),
            "brier_raw": float(brier_score_loss(y[te], raw_te)), "brier_platt": float(brier_score_loss(y[te], cal_te)),
            "brier_isotonic": float(brier_score_loss(y[te], iso_te)),
            "ece_raw": ece(y[te], raw_te), "ece_platt": ece(y[te], cal_te), "ece_isotonic": ece(y[te], iso_te),
        },
        "reliability_raw": rel(raw_te), "reliability_calibrated": rel(cal_te),
    })
    # routing table: what fraction goes to the senior queue at each threshold, and how good is it
    rows = []
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        sel = cal_te >= th
        rows.append({"threshold": th, "routed_share": float(sel.mean()),
                     "precision": float(y[te][sel].mean()) if sel.any() else None,
                     "recall": float(y[te][sel].sum() / max(y[te].sum(), 1))})
    out["routing"] = rows

    # --- explainability: TreeSHAP-style contributions (LightGBM pred_contrib); global top features
    contrib = gbm.booster_.predict(X[te], pred_contrib=True)[:, :-1]
    imp = np.abs(contrib).mean(0)
    out["top_features"] = [{"feature": names[i], "mean_abs_contrib": float(imp[i])} for i in np.argsort(-imp)[:12]]
    out["latency_b1"] = latency_ms(lambda b: gbm.predict_proba(np.vstack(b)), list(X[te]), 1, 300)

    ver = new_version()
    d = ART / "escalation" / ver
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"gbm": gbm, "platt": platt, "pca": pca, "names": names, "category_lr": cat_m, "priority": {
        "feat": feat, "scaler": scaler, "lr": pri_m}, "dedup_tau": DEDUP_TAU, "dedup_window_h": DEDUP_WINDOW_H}, d / "escalation.joblib")
    (d / "meta.json").write_text(json.dumps({"version": ver, "features": names, "threshold_senior": 0.5}))
    (ART / "escalation" / "CURRENT").write_text(ver)
    out["version"] = ver
    save_metrics("escalation", out)

    t = out["test"]
    print(f"test AUC {t['roc_auc']:.3f} (logreg baseline {out['logreg_baseline_auc']:.3f}) PR-AUC {t['pr_auc']:.3f}")
    print(f"Brier raw {t['brier_raw']:.4f} -> platt {t['brier_platt']:.4f} | iso {t['brier_isotonic']:.4f}")
    print(f"ECE raw {t['ece_raw']:.4f} -> platt {t['ece_platt']:.4f} | iso {t['ece_isotonic']:.4f}")
    print("top features:", [f["feature"] for f in out["top_features"][:6]])
    print("routing:", [(r["threshold"], round(r["routed_share"], 2), r["precision"] and round(r["precision"], 2), round(r["recall"], 2)) for r in rows])


if __name__ == "__main__":
    main()
