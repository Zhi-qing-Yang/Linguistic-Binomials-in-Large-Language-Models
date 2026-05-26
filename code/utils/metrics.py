import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def kl_bernoulli(p, q, eps=1e-12):
    p = min(max(float(p), eps), 1.0 - eps)
    q = min(max(float(q), eps), 1.0 - eps)
    return p * math.log(p / q, 2) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q), 2)


def jsd_bernoulli(p, q, eps=1e-12):
    m = 0.5 * (float(p) + float(q))
    return 0.5 * kl_bernoulli(p, m, eps) + 0.5 * kl_bernoulli(q, m, eps)


def compute_behavioral_metrics(
    data,
    include_mean_p_llm=False,
    include_n=False,
    nan_rho_to_zero=False,
):
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if len(df) < 2:
        return None

    p_llm = df["p_llm"].astype(float).to_numpy()
    p_corpus = df["p_corpus"].astype(float).to_numpy()
    preferred = df["preferred_order"].to_numpy()

    rho, _ = spearmanr(p_llm, p_corpus)
    if pd.isna(rho) and nan_rho_to_zero:
        rho = 0.0

    acc = float(((p_llm > 0.5) == (preferred == "A_B")).mean())
    mae = float(np.abs(p_llm - p_corpus).mean())
    jsd = float(np.mean([jsd_bernoulli(pl, pc) for pl, pc in zip(p_llm, p_corpus)]))

    metrics = {
        "rho": round(float(rho), 4),
        "acc": round(acc, 4),
        "mae": round(mae, 4),
        "jsd": round(jsd, 4),
    }

    if include_mean_p_llm:
        metrics["mean_p_llm"] = round(float(p_llm.mean()), 4)
    if include_n:
        metrics["n"] = int(len(df))

    return metrics
