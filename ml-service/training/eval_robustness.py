"""Honest-difficulty checks for the category classifiers (the synthetic data is easy; quantify how easy).

  A. mask error codes  : text with codes like BILL-CHG-010 / 403 replaced by <CODE> (codes leak the root cause)
  B. unseen root causes: hold out whole root causes (3 per category) - train never sees those issues
  C. both              : masked + unseen (hardest, closest to a genuinely new problem)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from app.features.text import ERROR_CODE
from training.common import METRICS, embed, load_tickets, save_metrics
from training.train_category import tfidf_pipeline

HELD_PER_CAT = 3


def mask(texts):
    return [ERROR_CODE.sub("<CODE>", t) for t in texts]


def fit_eval(name, Xtr_text, ytr, Xte_text, yte, Etr, Ete, out):
    t = tfidf_pipeline(3).fit(Xtr_text, ytr)
    e = LogisticRegression(C=30, max_iter=3000, class_weight="balanced").fit(Etr, ytr)
    r = {"tfidf_lr": float(f1_score(yte, t.predict(Xte_text), average="macro")),
         "emb_lr": float(f1_score(yte, e.predict(Ete), average="macro")), "n_train": len(ytr), "n_test": len(yte)}
    out[name] = r
    print(f"{name:32} tfidf {r['tfidf_lr']:.4f}  emb {r['emb_lr']:.4f}  (train {len(ytr)}, test {len(yte)})")


def main():
    df, E = load_tickets()
    y = df.category.values
    tr, te = (df.split == "train").values, (df.split == "test").values
    text, mtext = df.text.tolist(), mask(df.text.tolist())
    Em = embed(mtext)  # masked embeddings
    out = {}

    T = np.array(text, dtype=object)
    M = np.array(mtext, dtype=object)
    fit_eval("baseline (seen root causes)", T[tr], y[tr], T[te], y[te], E[tr], E[te], out)
    fit_eval("A. error codes masked", M[tr], y[tr], M[te], y[te], Em[tr], Em[te], out)

    rng = np.random.default_rng(3)
    held = []
    for c, g in df.groupby("category").root_cause_id.unique().items():
        held += list(rng.choice(g, HELD_PER_CAT, replace=False))
    is_held = df.root_cause_id.isin(held).values
    out["held_out_root_causes"] = [str(h) for h in held]
    fit_eval("B. unseen root causes", T[tr & ~is_held], y[tr & ~is_held], T[is_held], y[is_held],
             E[tr & ~is_held], E[is_held], out)
    # keep per-ticket predictions of the "unseen root cause" models for the cascade evaluation
    import pandas as pd
    t_b = tfidf_pipeline(3).fit(T[tr & ~is_held], y[tr & ~is_held])
    e_b = LogisticRegression(C=30, max_iter=3000, class_weight="balanced").fit(E[tr & ~is_held], y[tr & ~is_held])
    Pt, Pe = t_b.predict_proba(T[is_held]), e_b.predict_proba(E[is_held])
    pd.DataFrame({"ticket_uid": df.ticket_uid[is_held].values, "y": y[is_held],
                  "tfidf_pred": t_b.classes_[Pt.argmax(1)], "tfidf_conf": Pt.max(1),
                  "emb_pred": e_b.classes_[Pe.argmax(1)], "emb_conf": Pe.max(1)}
                 ).to_parquet(METRICS / "robust_preds.parquet")
    fit_eval("C. unseen + masked", M[tr & ~is_held], y[tr & ~is_held], M[is_held], y[is_held],
             Em[tr & ~is_held], Em[is_held], out)
    save_metrics("robustness", out)


if __name__ == "__main__":
    main()
