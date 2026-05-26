import argparse
from pathlib import Path

import pandas as pd

from utils.io import load_many_jsonl
from utils.metrics import compute_behavioral_metrics


def compute_metrics(df):
    return compute_behavioral_metrics(df)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more scored JSONL files to average over",
    )
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument(
        "--tier",
        choices=["strong", "medium", "weak"],
        default=None,
        help="Optional evidence tier filter",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_paths = [Path(x) for x in args.inputs]
    df = load_many_jsonl(input_paths)

    if "condition" in df.columns:
        df = df[df["condition"] == "canonical"].copy()

    if args.tier is not None:
        if "evidence_tier" not in df.columns:
            raise ValueError("Column 'evidence_tier' not found, but --tier was provided.")
        df = df[df["evidence_tier"] == args.tier].copy()

    model_name = df["model"].iloc[0] if "model" in df.columns and not df.empty else "unknown"
    langs = sorted(df["lang"].unique())

    rows = []
    for lang in langs:
        sl = df[df["lang"] == lang].copy()
        m = compute_metrics(sl)
        if m:
            rows.append({"lang": lang, **m})

    result = pd.DataFrame(rows)

    print("\nModel: {}".format(model_name))
    if args.tier is None:
        print("Main summary table")
    else:
        print("Main summary table ({})".format(args.tier))

    if not result.empty:
        print(result.to_string(index=False))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8")
    print("\n[DONE] Results saved to {}".format(out))


if __name__ == "__main__":
    main()