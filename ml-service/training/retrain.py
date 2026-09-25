"""Close the loop: retrain the category model from agent corrections, promote only if it does not regress.

  uv run python -m training.retrain --api http://localhost:8080 --ml http://localhost:8000 [--dry-run] [--min-new 20]

1. Export labels agents corrected in the dashboard (GET /api/admin/export/training-data, category_source = AGENT).
2. Train a candidate = original train set + 80% of the corrected tickets (20% held out as a "recent traffic" probe).
3. Promotion gate (all must hold, else the current model stays):
     - macro-F1 on the FROZEN test set >= current - 0.005
     - no class's F1 drops by more than 0.03
4. If promoted: write a new artifact version, move CURRENT, and hot-reload the running ML service.
Every run writes eval/metrics/retrain_<version>.json (what was compared, and the decision).
"""
import argparse
import json
import shutil

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from training.common import ART, load_tickets, new_version, save_metrics, embed

MAX_MACRO_DROP, MAX_CLASS_DROP = 0.005, 0.03


def fetch_corrections(api: str, user: str, password: str) -> pd.DataFrame:
    with httpx.Client(base_url=api, timeout=120) as c:
        tok = c.post("/api/auth/login", json={"username": user, "password": password}).raise_for_status().json()["accessToken"]
        r = c.get("/api/admin/export/training-data", headers={"Authorization": f"Bearer {tok}"}).raise_for_status()
    rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df.category_source == "AGENT"].drop_duplicates("id").reset_index(drop=True)


def per_class_f1(y, pred, classes):
    return {c: float(f1_score(y, pred, labels=[c], average="macro", zero_division=0)) for c in classes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--ml", default=None, help="ML service URL to hot-reload after promotion")
    ap.add_argument("--admin-user", default="admin")
    ap.add_argument("--admin-password", default="admin")
    ap.add_argument("--admin-key", default="dev-admin-key")
    ap.add_argument("--min-new", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fb = fetch_corrections(a.api, a.admin_user, a.admin_password)
    print(f"agent-corrected tickets available: {len(fb)}")
    if len(fb) < a.min_new:
        print(f"not enough new labels (need >= {a.min_new}); nothing to do")
        return

    cur_ver = (ART / "category" / "CURRENT").read_text().strip()
    cur_dir = ART / "category" / cur_ver
    current = joblib.load(cur_dir / "emb_lr.joblib")
    meta = json.loads((cur_dir / "meta.json").read_text())

    df, E = load_tickets()
    tr, te = (df.split == "train").values, (df.split == "test").values
    fb["text"] = fb.subject + "\n" + fb.body
    Efb = embed(fb.text.tolist())
    rng = np.random.default_rng(0)
    hold = rng.random(len(fb)) < 0.2
    X = np.vstack([E[tr], Efb[~hold]])
    y = np.concatenate([df.category[tr].values, fb.category.values[~hold]])
    cand = LogisticRegression(C=meta["emb_C"], max_iter=3000, class_weight="balanced").fit(X, y)

    classes = list(current.classes_)
    yt = df.category[te].values
    f_cur, f_new = f1_score(yt, current.predict(E[te]), average="macro"), f1_score(yt, cand.predict(E[te]), average="macro")
    pc_cur, pc_new = per_class_f1(yt, current.predict(E[te]), classes), per_class_f1(yt, cand.predict(E[te]), classes)
    worst = max(pc_cur[c] - pc_new[c] for c in classes)
    probe = {"n": int(hold.sum()),
             "current_accuracy": float(accuracy_score(fb.category[hold], current.predict(Efb[hold]))) if hold.any() else None,
             "candidate_accuracy": float(accuracy_score(fb.category[hold], cand.predict(Efb[hold]))) if hold.any() else None}
    checks = {"macro_f1_ok": bool(f_new >= f_cur - MAX_MACRO_DROP), "per_class_ok": bool(worst <= MAX_CLASS_DROP)}
    promote = all(checks.values())
    ver = new_version()
    report = {"new_version": ver, "current_version": cur_ver, "corrections_used": int((~hold).sum()), "frozen_test": {
        "current_macro_f1": float(f_cur), "candidate_macro_f1": float(f_new), "worst_class_drop": float(worst)},
        "recent_traffic_probe": probe, "gate": checks, "promoted": bool(promote and not a.dry_run)}
    print(json.dumps(report, indent=2))

    if promote and not a.dry_run:
        d = ART / "category" / ver
        d.mkdir(parents=True)
        for f in ["tfidf_lr.joblib", "meta.json"]:
            shutil.copy(cur_dir / f, d / f)
        joblib.dump(cand, d / "emb_lr.joblib")
        meta.update({"version": ver, "parent": cur_ver, "retrained_on_corrections": int((~hold).sum())})
        (d / "meta.json").write_text(json.dumps(meta))
        (ART / "category" / "CURRENT").write_text(ver)
        print(f"PROMOTED {cur_ver} -> {ver}")
        if a.ml:
            r = httpx.post(f"{a.ml}/admin/reload", headers={"x-admin-key": a.admin_key}, timeout=120)
            print("ML service reload:", r.status_code, r.json().get("versions", {}).get("category"))
    else:
        print("NOT promoted" + (" (dry run)" if a.dry_run else ": gate failed, current model kept"))
    save_metrics(f"retrain_{ver}", report)


if __name__ == "__main__":
    main()
