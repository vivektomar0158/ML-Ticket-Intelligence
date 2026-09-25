"""Redact personal data before any text leaves for an external LLM."""
import re

_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]?\d{3,4}[ .-]?\d{3,4}(?!\w)")),
    ("SECRET", re.compile(r"\b(?:sk|pk|api|key|tok|ghp|AIza)[-_A-Za-z0-9]{16,}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
]


def redact(text: str) -> tuple[str, int]:
    """Return (redacted text, number of redactions)."""
    n = 0
    for label, rx in _PATTERNS:
        text, k = rx.subn(f"[{label}]", text)
        n += k
    return text, n
