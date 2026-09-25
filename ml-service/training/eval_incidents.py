"""Replay the timeline through the production dedup/cluster/incident logic and measure what operators care about:
  - incident recall + detection delay (minutes and #tickets seen before the alarm)
  - false incident alarms (flagged clusters that are not a real incident)
  - cluster purity
Logic mirrors the Java DuplicateService: link a new ticket to earlier tickets (same product, within `window`)
with cosine >= tau, union their clusters; flag an incident when >= MIN_TICKETS members arrived within 60 min.
"""
import numpy as np
import pandas as pd

from training.common import epoch_s, load_tickets, save_metrics

MIN_TICKETS, MIN_CUSTOMERS, BURST_S = 5, 3, 3600


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x


def simulate(df, E, tau, window_h):
    """Returns alarms = [(t_flag, snapshot_member_indices)]. A cluster's *live* members are those from the last 60 min."""
    n = len(df)
    order = np.argsort(df.created_at.values)
    ts = epoch_s(df.created_at)
    prod, cust = df["product"].values, df.customer_id.values
    uf, members = UF(n), {i: [i] for i in range(n)}
    alarms = []
    for pos, i in enumerate(order):
        lo = np.searchsorted(ts[order], ts[i] - window_h * 3600)
        cand = order[lo:pos]
        cand = cand[prod[cand] == prod[i]]
        if len(cand):
            s = E[cand] @ E[i]
            top = np.argsort(-s)[:20]
            for j in cand[top][s[top] >= tau]:
                a, b = uf.find(i), uf.find(j)
                if a != b:
                    uf.p[b] = a
                    members[a] += members.pop(b)
        live = [m for m in members[uf.find(i)] if ts[m] >= ts[i] - BURST_S]
        if len(live) >= MIN_TICKETS and len({cust[m] for m in live}) >= MIN_CUSTOMERS:
            ls = set(live)
            # one alarm per burst: skip if a recent alarm already covers >= half of these tickets
            if not any(ts[i] - t <= BURST_S and len(ls & snap) >= 0.5 * len(ls) for t, snap in alarms[-50:]):
                alarms.append((ts[i], ls))
    return alarms, ts


def evaluate(df, E, tau, window_h, lo_day=None, hi_day=None):
    sub = df
    if lo_day is not None:
        d = (df.created_at - df.created_at.min()).dt.days
        sub = df[(d >= lo_day) & (d < hi_day)]
    idx = sub.index.values
    sdf = sub.reset_index(drop=True)
    alarms, ts = simulate(sdf, E[idx], tau, window_h)
    inc = sdf.incident_id.values
    true_inc = list(pd.unique(inc[pd.notna(inc)]))
    false_alarms, purities = 0, []
    for t, snap in alarms:
        labs = pd.Series(inc[list(snap)]).fillna("none").value_counts()
        purities.append(labs.iloc[0] / len(snap))
        if labs.index[0] == "none" or labs.iloc[0] / len(snap) < 0.5:
            false_alarms += 1
    delays, before, detected = [], [], 0
    for k in true_inc:
        ii = set(np.where(inc == k)[0])
        first = min(ts[list(ii)])
        hits = [(t, snap) for t, snap in alarms if len(snap & ii) >= MIN_TICKETS]
        if hits:
            t_flag = min(t for t, _ in hits)
            detected += 1
            delays.append((t_flag - first) / 60)
            before.append(int(sum(1 for m in ii if ts[m] <= t_flag)))
    days = max((sdf.created_at.max() - sdf.created_at.min()).days, 1)
    return {"tau": tau, "window_h": window_h, "incidents": len(true_inc), "detected": detected,
            "recall": detected / max(len(true_inc), 1), "median_delay_min": float(np.median(delays)) if delays else None,
            "median_tickets_before_alarm": float(np.median(before)) if before else None,
            "alarms": len(alarms), "false_alarms": false_alarms, "false_alarms_per_week": false_alarms / days * 7,
            "mean_alarm_purity": float(np.mean(purities)) if purities else None}


def main():
    df, E = load_tickets()
    df = df.sort_values("created_at").reset_index(drop=True)  # E must stay aligned
    order = load_tickets()[0].sort_values("created_at").index.values
    E = E[order]
    rows = []
    for w in [3, 72]:
        for tau in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            r = evaluate(df, E, tau, w, 0, 72)  # tune on train+val period
            rows.append(r)
            print(f"window {w:>2}h tau {tau:.2f}: recall {r['recall']:.2f} delay {r['median_delay_min']} "
                  f"false alarms/wk {r['false_alarms_per_week']:.2f} purity {r['mean_alarm_purity'] and round(r['mean_alarm_purity'], 2)}")
    ok = [r for r in rows if r["false_alarms_per_week"] <= 1.0] or rows
    best = max(ok, key=lambda r: (r["recall"], -(r["median_delay_min"] or 1e9)))
    test = evaluate(df, E, best["tau"], best["window_h"], 72, 90)
    full = evaluate(df, E, best["tau"], best["window_h"])
    print("chosen:", best["tau"], best["window_h"], "| test period:", test)
    save_metrics("incidents", {"sweep_tuning_period_days_0_72": rows, "chosen": {"tau": best["tau"], "window_h": best["window_h"]},
                               "test_period_days_72_90": test, "full_90_days": full})


if __name__ == "__main__":
    main()
