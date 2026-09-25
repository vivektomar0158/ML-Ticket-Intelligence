"""Priority feature construction shared by training and serving."""
import numpy as np
from scipy.sparse import csr_matrix, hstack

from app.features.text import extract_matrix

TIERS = ["FREE", "PRO", "ENTERPRISE"]


def tier_onehot(tiers) -> np.ndarray:
    return np.array([[t == k for k in TIERS] for t in tiers], dtype=np.float32)


def build(feat, scaler, texts, tiers, fit: bool = False):
    """TF-IDF features + standardised urgency signals + tier one-hot."""
    texts = list(texts)
    S = extract_matrix(texts)
    S = scaler.fit_transform(S) if fit else scaler.transform(S)
    Xt = feat.fit_transform(texts) if fit else feat.transform(texts)
    return hstack([Xt, csr_matrix(S), csr_matrix(tier_onehot(tiers))]).tocsr()
