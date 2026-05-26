import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coef", required=True, help="Path to coefficient CSV")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--rep_type", default="last")
    p.add_argument("--hidden_dim", type=int, default=2560)
    p.add_argument("--output", required=True, help="Output .npy path")
    p.add_argument("--stable_only", action="store_true")
    p.add_argument("--n_splits", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    coef_path = Path(args.coef)
    if not coef_path.exists():
        raise FileNotFoundError(coef_path)

    df = pd.read_csv(coef_path)

    required = {"layer", "rep_type", "dim", "fold", "coef"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in coef file: {missing}")

    sub = df[
        (df["layer"] == args.layer)
        & (df["rep_type"] == args.rep_type)
    ].copy()

    if sub.empty:
        raise ValueError(
            f"No rows found for layer={args.layer}, rep_type={args.rep_type}"
        )

    sub["dim"] = pd.to_numeric(sub["dim"], errors="coerce")
    sub["coef"] = pd.to_numeric(sub["coef"], errors="coerce")
    sub["fold"] = pd.to_numeric(sub["fold"], errors="coerce")
    sub = sub.dropna(subset=["dim", "coef", "fold"]).copy()

    sub["dim"] = sub["dim"].astype(int)
    sub["fold"] = sub["fold"].astype(int)

    max_dim = int(sub["dim"].max())
    if max_dim >= args.hidden_dim:
        raise ValueError(
            f"coef max dim={max_dim}, but hidden_dim={args.hidden_dim}"
        )

    print(f"[INFO] coef file     : {coef_path}")
    print(f"[INFO] layer         : {args.layer}")
    print(f"[INFO] rep_type      : {args.rep_type}")
    print(f"[INFO] hidden_dim    : {args.hidden_dim}")
    print(f"[INFO] rows selected : {len(sub)}")

    if args.stable_only:
        fold_count = sub.groupby("dim")["fold"].nunique()
        stable_dims = fold_count[fold_count == args.n_splits].index
        sub = sub[sub["dim"].isin(stable_dims)].copy()
        print(f"[INFO] stable dims   : {len(stable_dims)}")
    else:
        print("[INFO] mode          : all nonzero dims, averaged across folds")

    if sub.empty:
        raise ValueError("No coefficients left after filtering.")

    mean_coef = sub.groupby("dim")["coef"].mean()

    vec = np.zeros(args.hidden_dim, dtype=np.float32)
    for dim, val in mean_coef.items():
        vec[int(dim)] = float(val)

    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("Steering vector is all zeros.")

    vec = vec / norm

    n_nonzero = int((vec != 0).sum())
    print(f"[INFO] final nonzero dims: {n_nonzero}")
    print(f"[INFO] norm before normalization: {norm:.6f}")

    print("[INFO] top-10 dims by |mean coef|:")
    top = mean_coef.abs().sort_values(ascending=False).head(10)
    for dim, abs_val in top.items():
        print(f"  dim={int(dim):5d}  mean_coef={mean_coef[dim]:+.6f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vec.astype(np.float32))

    print(f"[DONE] saved to {out_path}")
    print(f"[DONE] shape: {vec.shape}")


if __name__ == "__main__":
    main()