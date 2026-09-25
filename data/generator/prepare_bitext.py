"""Step 4: prepare Bitext (a) as an external benchmark on its own labels, (b) mapped subset for augmentation."""
import pandas as pd

from data.generator.llm import ROOT

MAP = {  # Bitext intent -> CloudDesk category (unmapped intents are e-commerce only and dropped)
    "check_invoice": "BILLING", "get_invoice": "BILLING", "payment_issue": "BILLING", "check_payment_methods": "BILLING",
    "get_refund": "BILLING", "check_refund_policy": "BILLING", "track_refund": "BILLING", "check_cancellation_fee": "BILLING",
    "recover_password": "LOGIN_ACCESS", "registration_problems": "LOGIN_ACCESS",
    "create_account": "ACCOUNT_MANAGEMENT", "delete_account": "ACCOUNT_MANAGEMENT",
    "edit_account": "ACCOUNT_MANAGEMENT", "switch_account": "ACCOUNT_MANAGEMENT",
}


def main():
    d = pd.read_csv(ROOT / "data/raw/bitext/bitext.csv")
    d = d.rename(columns={"instruction": "text"})[["text", "category", "intent"]]
    d["text"] = d.text.str.strip()
    d = d.drop_duplicates("text").reset_index(drop=True)

    # stratified random split; utterances are template-generated, so this benchmark is an easy one (documented)
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    parts = {"train": [], "val": [], "test": []}
    for _, g in d.groupby("category"):
        g = g.sample(frac=1, random_state=42)
        n = len(g)
        parts["train"].append(g.iloc[: int(.7 * n)])
        parts["val"].append(g.iloc[int(.7 * n): int(.85 * n)])
        parts["test"].append(g.iloc[int(.85 * n):])
    for k, v in parts.items():
        pd.concat(v).to_parquet(out / f"bitext_{k}.parquet")
        print(k, sum(len(x) for x in v))

    m = d[d.intent.isin(MAP)].copy()
    m["category"] = m.intent.map(MAP)
    m.to_parquet(out / "bitext_mapped.parquet")
    print("mapped for augmentation:", m.category.value_counts().to_dict())


if __name__ == "__main__":
    main()
