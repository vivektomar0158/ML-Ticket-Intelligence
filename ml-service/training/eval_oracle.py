"""Ceiling for the escalation task: a model that sees the *true hidden generator inputs* (tone, deadline, incident, ...)."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from training.common import load_tickets, save_metrics


def main():
    df, _ = load_tickets()
    X = pd.DataFrame({"tier": df.customer_tier.map({"FREE": 0, "PRO": .5, "ENTERPRISE": 1.2}),
                      "hi": df.priority.isin(["HIGH", "URGENT"]).astype(int), "urg": (df.priority == "URGENT").astype(int),
                      "neg": df.tone.map({"angry": 1, "frustrated": .6, "terse": .2}).fillna(0),
                      "dl": df.has_deadline.astype(int), "inc": df.incident_id.notna().astype(int),
                      "off": df.off_hours.astype(int), "prior": df.prior_tickets_7d.clip(upper=5),
                      "priv": (df.category == "DATA_PRIVACY").astype(int)})
    tr, te = (df.split == "train").values, (df.split == "test").values
    m = LogisticRegression(C=10, max_iter=2000).fit(X[tr], df.escalated[tr])
    auc = float(roc_auc_score(df.escalated[te], m.predict_proba(X[te])[:, 1]))
    save_metrics("escalation_oracle", {"oracle_auc": auc})
    print("oracle AUC:", round(auc, 3))


if __name__ == "__main__":
    main()
