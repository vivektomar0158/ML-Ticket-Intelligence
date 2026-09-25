"""Priority model: TF-IDF + urgency signals + tier -> multinomial LR, URGENT decision weight tuned for recall.

  uv run python -m training.train_priority
"""
import json

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from app.features.text import extract_matrix
from training.common import ART, PRIORITIES, latency_ms, load_tickets, multiclass_ece, new_version, save_metrics
from training.train_category import tfidf_pipeline

TIERS = ["FREE", "PRO", "ENTERPRISE"]


def tier_onehot(tiers):
    return np.array([[t == k for k in TIERS] for t in tiers], dtype=np.float32)


def build(feat, scaler, texts, tiers, fit=False):
    S = extract_matrix(list(texts))
    S = scaler.fit_transform(S) if fit else scaler.transform(S)
    Xt = feat.fit_transform(list(texts)) if fit else feat.transform(list(texts))
    return hstack([Xt, csr_matrix(S), csr_matrix(tier_onehot(tiers))]).tocsr()


def urgent_metrics(y_idx, pred):
    u = PRIORITIES.index("URGENT")
    return {"urgent_recall": float(((pred == u) & (y_idx == u)).sum() / max((y_idx == u).sum(), 1)),
            "urgent_precision": float(((pred == u) & (y_idx == u)).sum() / max((pred == u).sum(), 1)),
            "under_prioritized_2plus": float(((y_idx - pred) >= 2).mean()),
            "macro_f1": float(f1_score(y_idx, pred, average="macro")), "accuracy": float(accuracy_score(y_idx, pred))}


def main():
    df, _ = load_tickets()
    tr, va, te = (df.split == "train").values, (df.split == "val").values, (df.split == "test").values
    y = df.priority.map(PRIORITIES.index).values
    feat = tfidf_pipeline(1).named_steps["f"]
    scaler = StandardScaler()
    Xtr = build(feat, scaler, df.text[tr], df.customer_tier[tr], fit=True)
    Xva = build(feat, scaler, df.text[va], df.customer_tier[va])
    Xte = build(feat, scaler, df.text[te], df.customer_tier[te])

    best = None
    for C in [0.3, 1, 3, 10]:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr])
        f = f1_score(y[va], m.predict(Xva), average="macro")
        print(f"priority LR C={C}: val macro-F1 {f:.4f}")
        if best is None or f > best[0]:
            best = (f, C, m)
    _, C, lr = best

    # ablation: without the hand-crafted signals / tier (does feature engineering help?)
    n_text = Xtr.shape[1] - 12 - 3
    ab = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr[:, :n_text], y[tr])
    ablation = float(f1_score(y[te], ab.predict(Xte[:, :n_text]), average="macro"))

    # tune the URGENT weight on val: smallest w with val URGENT recall >= 0.90 (keeps precision as high as possible)
    u = PRIORITIES.index("URGENT")
    Pva = lr.predict_proba(Xva)
    w_best = 1.0
    for w in np.arange(1.0, 6.01, 0.25):
        adj = Pva * np.array([1, 1, 1, w])
        if urgent_metrics(y[va], adj.argmax(1))["urgent_recall"] >= 0.90:
            w_best = float(w)
            break
    w_vec = np.array([1, 1, 1, w_best])
    Pte = lr.predict_proba(Xte)
    out = {"C": C, "urgent_weight": w_best, "val_macro_f1": best[0], "text_only_macro_f1": ablation,
           "default": urgent_metrics(y[te], Pte.argmax(1)),
           "tuned_for_urgent_recall": urgent_metrics(y[te], (Pte * w_vec).argmax(1)),
           "ece": multiclass_ece(y[te], Pte),
           "per_class": classification_report(y[te], Pte.argmax(1), target_names=PRIORITIES, output_dict=True, zero_division=0),
           "confusion_tuned": confusion_matrix(y[te], (Pte * w_vec).argmax(1)).tolist()}

    texts, tiers = df.text[te].tolist(), df.customer_tier[te].tolist()
    pairs = list(zip(texts, tiers))
    out["latency_b1"] = latency_ms(lambda b: lr.predict_proba(build(feat, scaler, [x[0] for x in b], [x[1] for x in b])), pairs, 1, 200)

    ver = new_version()
    d = ART / "priority" / ver
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"feat": feat, "scaler": scaler, "lr": lr, "urgent_weight": w_best, "tiers": TIERS}, d / "priority.joblib")
    (d / "meta.json").write_text(json.dumps({"version": ver, "classes": PRIORITIES, "urgent_weight": w_best}))
    (ART / "priority" / "CURRENT").write_text(ver)
    out["version"] = ver
    save_metrics("priority", out)
    print("default :", {k: round(v, 3) for k, v in out["default"].items()})
    print("tuned   :", {k: round(v, 3) for k, v in out["tuned_for_urgent_recall"].items()}, "w =", w_best)
    print("text-only macro-F1:", round(ablation, 4), "| with signals+tier:", round(out["default"]["macro_f1"], 4))


if __name__ == "__main__":
    main()
