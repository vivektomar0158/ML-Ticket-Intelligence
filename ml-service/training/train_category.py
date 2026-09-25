"""Category classification: TF-IDF+LR vs MiniLM-embedding+LR (vs MLP). Also Bitext benchmark + augmentation test.

  uv run python -m training.train_category
Writes artifacts/category/<version>/{tfidf_lr.joblib, emb_lr.joblib, meta.json} + CURRENT, eval/metrics/category.json
and eval/metrics/category_preds.parquet (per-ticket predictions reused by the cascade evaluation).
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from training.common import (ART, CATEGORIES, DATA, embed, latency_ms, load_tickets, multiclass_ece, new_version,
                             save_metrics)


def tfidf_pipeline(C: float) -> Pipeline:
    feats = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=150_000)),
    ])
    return Pipeline([("f", feats), ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])


def report(y_true, proba, classes) -> dict:
    y_idx = np.array([classes.index(c) for c in y_true])
    pred = proba.argmax(1)
    return {
        "macro_f1": float(f1_score(y_idx, pred, average="macro")),
        "accuracy": float(accuracy_score(y_idx, pred)),
        "ece": multiclass_ece(y_idx, proba),
        "per_class": classification_report(y_idx, pred, target_names=classes, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_idx, pred).tolist(),
    }


def main():
    df, E = load_tickets()
    tr, va, te = (df.split == "train").values, (df.split == "val").values, (df.split == "test").values
    y = df.category.values
    out = {"classes": CATEGORIES, "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}}

    # ---- A. TF-IDF + LR (tune C on val)
    best = None
    for C in [1, 3, 10, 30]:
        p = tfidf_pipeline(C).fit(df.text[tr], y[tr])
        f = f1_score(y[va], p.predict(df.text[va]), average="macro")
        print(f"tfidf C={C}: val macro-F1 {f:.4f}")
        if best is None or f > best[0]:
            best = (f, C)
    tfidf = tfidf_pipeline(best[1]).fit(df.text[tr], y[tr])
    assert list(tfidf.classes_) == sorted(CATEGORIES)
    cls = list(tfidf.classes_)
    out["tfidf_lr"] = {"C": best[1], "val_macro_f1": best[0], **report(y[te], tfidf.predict_proba(df.text[te]), cls)}
    te_texts = df.text[te].tolist()
    out["tfidf_lr"]["latency_b1"] = latency_ms(lambda t: tfidf.predict_proba(t), te_texts, 1, 300)
    out["tfidf_lr"]["latency_b32"] = latency_ms(lambda t: tfidf.predict_proba(t), te_texts, 32, 40)

    # ---- B. embedding + LR / MLP
    best = None
    for C in [1, 3, 10, 30, 100]:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(E[tr], y[tr])
        f = f1_score(y[va], m.predict(E[va]), average="macro")
        print(f"emb LR C={C}: val macro-F1 {f:.4f}")
        if best is None or f > best[0]:
            best = (f, C)
    emb_lr = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(E[tr], y[tr])
    out["emb_lr"] = {"C": best[1], "val_macro_f1": best[0], **report(y[te], emb_lr.predict_proba(E[te]), cls)}
    mlp = MLPClassifier((256,), early_stopping=True, max_iter=200, random_state=0).fit(E[tr], y[tr])
    out["emb_mlp"] = {"val_macro_f1": float(f1_score(y[va], mlp.predict(E[va]), average="macro")),
                      **report(y[te], mlp.predict_proba(E[te]), cls)}
    # end-to-end latency includes embedding the text (what serving actually pays)
    out["emb_lr"]["latency_b1"] = latency_ms(lambda t: emb_lr.predict_proba(embed(t)), te_texts, 1, 200)
    out["emb_lr"]["latency_b32"] = latency_ms(lambda t: emb_lr.predict_proba(embed(t)), te_texts, 32, 20)

    # ---- augmentation experiment: add Bitext-mapped utterances to the training set
    bm = pd.read_parquet(DATA / "bitext_mapped.parquet").sample(3000, random_state=0)
    Ea = np.vstack([E[tr], embed(bm.text.tolist())])
    ya = np.concatenate([y[tr], bm.category.values])
    aug = LogisticRegression(C=out["emb_lr"]["C"], max_iter=3000, class_weight="balanced").fit(Ea, ya)
    fa = f1_score(y[te], aug.predict(E[te]), average="macro")
    out["augmentation"] = {"bitext_added": len(bm), "macro_f1_with": float(fa),
                           "macro_f1_without": out["emb_lr"]["macro_f1"],
                           "kept": bool(fa > out["emb_lr"]["macro_f1"] + 0.002)}
    print("augmentation:", out["augmentation"])

    # ---- Bitext external benchmark (its own 11 categories)
    b = {k: pd.read_parquet(DATA / f"bitext_{k}.parquet") for k in ["train", "val", "test"]}
    bcls = sorted(b["train"].category.unique())
    bt = tfidf_pipeline(3).fit(b["train"].text, b["train"].category)
    Eb = {k: embed(v.text.tolist()) for k, v in b.items()}
    be = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(Eb["train"], b["train"].category)
    out["bitext"] = {
        "classes": bcls, "n_test": len(b["test"]),
        "tfidf_lr_macro_f1": float(f1_score(b["test"].category, bt.predict(b["test"].text), average="macro")),
        "emb_lr_macro_f1": float(f1_score(b["test"].category, be.predict(Eb["test"]), average="macro")),
        "note": "Bitext utterances are template-generated (~9 words); this benchmark is easy and saturates.",
    }
    print("bitext:", {k: v for k, v in out["bitext"].items() if "f1" in k})

    # ---- persist artifacts + per-ticket predictions (val+test) for cascade analysis
    ver = new_version()
    d = ART / "category" / ver
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, d / "tfidf_lr.joblib")
    joblib.dump(emb_lr, d / "emb_lr.joblib")
    (d / "meta.json").write_text(json.dumps({"version": ver, "classes": cls, "embedding": "minilm-l6-v2@1",
                                             "tfidf_C": out["tfidf_lr"]["C"], "emb_C": out["emb_lr"]["C"]}))
    (ART / "category" / "CURRENT").write_text(ver)
    out["version"] = ver

    P_t, P_e = tfidf.predict_proba(df.text), emb_lr.predict_proba(E)
    preds = pd.DataFrame({"ticket_uid": df.ticket_uid, "split": df.split, "y": y,
                          "tfidf_pred": np.array(cls)[P_t.argmax(1)], "tfidf_conf": P_t.max(1),
                          "emb_pred": np.array(cls)[P_e.argmax(1)], "emb_conf": P_e.max(1)})
    preds.to_parquet(ART.parent.parent / "eval" / "metrics" / "category_preds.parquet")
    save_metrics("category", out)
    for k in ["tfidf_lr", "emb_lr", "emb_mlp"]:
        print(f"{k:9} test macro-F1 {out[k]['macro_f1']:.4f} acc {out[k]['accuracy']:.4f} ECE {out[k]['ece']:.3f}")
    print("latency tfidf b1:", out["tfidf_lr"]["latency_b1"]["p50_ms"], "emb b1:", out["emb_lr"]["latency_b1"]["p50_ms"])


if __name__ == "__main__":
    main()
