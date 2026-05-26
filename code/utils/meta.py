from pathlib import Path

import pandas as pd


def load_meta(lang: str, data_dir) -> dict[int, dict]:
    data_dir = Path(data_dir)
    path = data_dir / lang / f"output_{lang}_sk.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["Binomial_ID"] = pd.to_numeric(df["Binomial_ID"], errors="coerce")
    df = df.dropna(subset=["Binomial_ID"]).copy()
    df["Binomial_ID"] = df["Binomial_ID"].astype(int)

    required = [
        f"A_{lang}",
        f"B_{lang}",
        "A_B_count",
        "B_A_count",
        "p_corpus",
        "evidence_tier",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    meta = {}
    for _, row in df.iterrows():
        binom_id = int(row["Binomial_ID"])

        conj = row.get("Preferred_conjunction", "")
        conj = "" if pd.isna(conj) else str(conj).strip()

        ab = float(row["A_B_count"])
        ba = float(row["B_A_count"])

        meta[binom_id] = {
            "a": str(row[f"A_{lang}"]).strip(),
            "b": str(row[f"B_{lang}"]).strip(),
            "conj": conj,
            "p_corpus": float(row["p_corpus"]),
            "evidence_tier": str(row["evidence_tier"]),
            "preferred_order": "A_B" if ab >= ba else "B_A",
        }

    return meta
