import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

LANGS = ["en", "de", "ru", "zh", "ja", "tr", "id", "ar"]

ALPHA = 0.5
TAU_LO_PCT = 33.0
TAU_HI_PCT = 67.0

COL_AB = "A_B_count"
COL_BA = "B_A_count"


def assign_tier(n: float, tau_lo: float, tau_hi: float) -> str:
    if n >= tau_hi:
        return "strong"
    elif n >= tau_lo:
        return "medium"
    else:
        return "weak"


def process_lang(lang: str):
    csv_path = DATA_DIR / lang / f"output_{lang}_sk.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    for col in [COL_AB, COL_BA]:
        if col not in df.columns:
            print(f"[ERROR] Column '{col}' missing in {csv_path}")
            return

    df[COL_AB] = pd.to_numeric(df[COL_AB], errors="coerce").fillna(0).astype(int)
    df[COL_BA] = pd.to_numeric(df[COL_BA], errors="coerce").fillna(0).astype(int)

    df["p_corpus"] = (
        (df[COL_AB] + ALPHA) / (df[COL_AB] + df[COL_BA] + 2 * ALPHA)
    ).round(4)

    n = df[COL_AB] + df[COL_BA]
    tau_lo = float(np.percentile(n, TAU_LO_PCT))
    tau_hi = float(np.percentile(n, TAU_HI_PCT))

    print(
        f"[{lang.upper()}] tau_lo={tau_lo:.1f}  tau_hi={tau_hi:.1f}  "
        f"(n range: {n.min():.0f}–{n.max():.0f})"
    )

    df["evidence_tier"] = n.apply(lambda x: assign_tier(x, tau_lo, tau_hi))

    tier_counts = df["evidence_tier"].value_counts()
    print(f"         tiers → {dict(tier_counts)}")

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[DONE]   {csv_path}")


def main():
    for lang in LANGS:
        process_lang(lang)
    print("\nAll languages processed.")


if __name__ == "__main__":
    main()