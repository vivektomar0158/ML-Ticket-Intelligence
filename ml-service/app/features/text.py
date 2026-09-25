"""Hand-crafted text signals shared by training and serving (single source of truth)."""
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

DEADLINE = re.compile(
    r"\b(asap|immediately|right now|urgent(ly)?|deadline|by (eod|end of (the )?(day|week))|"
    r"(by|before|until|due|within)\s+(the\s+)?(\d{1,2}(:\d{2})?\s?(am|pm)|today|tonight|tomorrow|monday|tuesday|wednesday|"
    r"thursday|friday|noon|midnight|the weekend|\d+\s+(hours?|minutes?))|"
    r"(demo|audit|payroll|launch|presentation|board meeting|renewal|go-live|release)\b.{0,30}\b(today|tonight|tomorrow|friday|monday|\d{1,2}\s?(am|pm)))",
    re.I)
OUTAGE = re.compile(
    r"\b(outage|is down|are down|went down|not loading for (everyone|all)|all (of )?(our )?users|everyone|entire (team|company|org)|"
    r"whole (team|company|office)|nobody|no one can|production|prod\b|multiple users|company-wide|"
    r"since (the|your) (update|release|deploy)|after (the|your) (update|release|upgrade))",
    re.I)
ERROR_CODE = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9.]+)+|[45]\d\d)\b")
BLOCKED = re.compile(r"\b(can'?t|cannot|unable to|blocked|locked out|stuck|not working|broken|fails?|failing|crash(es|ed|ing)?)\b", re.I)
LEGAL = re.compile(r"\b(gdpr|breach|compromised|unauthori[sz]ed|hacked|leak(ed)?|lawyer|legal|lawsuit|chargeback)\b", re.I)
CANCEL = re.compile(r"\b(cancel(l?ing)?( our)? (subscription|contract|account|plan)|switch(ing)? to (a )?(competitor|another)|refund)\b", re.I)

SIGNAL_NAMES = ["deadline", "outage", "error_code", "blocked", "legal", "churn"]
FEATURE_NAMES = ["sig_deadline", "sig_outage", "sig_error_code", "sig_blocked", "sig_legal", "sig_churn",
                 "exclaim", "caps_ratio", "questions", "n_words", "vader_neg", "vader_compound"]


def extract(text: str) -> tuple[list[float], list[str]]:
    """Return (numeric feature vector in FEATURE_NAMES order, human-readable signals that fired)."""
    words = text.split()
    n = max(len(words), 1)
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / max(len(letters), 1)
    hits = {
        "deadline": bool(DEADLINE.search(text)),
        "outage": bool(OUTAGE.search(text)),
        "error_code": bool(ERROR_CODE.search(text)),
        "blocked": bool(BLOCKED.search(text)),
        "legal": bool(LEGAL.search(text)),
        "churn": bool(CANCEL.search(text)),
    }
    v = _vader.polarity_scores(text)
    vec = [float(hits[k]) for k in SIGNAL_NAMES] + [
        float(min(text.count("!"), 5)), caps, float(min(text.count("?"), 5)), float(min(n, 300)) / 300.0,
        v["neg"], v["compound"],
    ]
    signals = [k for k, h in hits.items() if h]
    m = DEADLINE.search(text)
    if m:
        signals = ["deadline:" + m.group(0).strip()[:30] if s == "deadline" else s for s in signals]
    return vec, signals


def extract_matrix(texts: list[str]):
    import numpy as np

    return np.array([extract(t)[0] for t in texts], dtype=np.float32)
