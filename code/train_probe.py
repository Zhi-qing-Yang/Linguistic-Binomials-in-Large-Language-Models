import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.io import load_jsonl
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to reps.jsonl")
    parser.add_argument("--output", required=True, help="Path to output summary csv")
    parser.add_argument("--coef_output", default=None, help="Optional path to save coefficients; only used in --mode cv")
    parser.add_argument("--mode", choices=["cv", "crosslingual"], default="cv")
    parser.add_argument(
        "--xling_split",
        choices=["binom_id", "language"],
        default="binom_id",
        help=(
            "Only used in --mode crosslingual. "
            "'binom_id' uses held-out binomial IDs for every src->tgt pair; "
            "'language' trains on all source-language rows and tests on all target-language rows."
        ),
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=3e-2)
    parser.add_argument("--l1_ratio", type=float, default=1.0)

    parser.add_argument("--rep_types", nargs="+", default=None)
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--tier", choices=["strong", "medium", "weak"], default=None)

    return parser.parse_args()


def safe_spearman(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    if pd.isna(rho):
        return 0.0
    return float(rho)


def build_groups(df):
    return (df["lang"].astype(str) + "_" + df["binom_id"].astype(str)).values


def build_pairwise_target(df):
    p = df["p_corpus"].astype(float).values
    return np.abs(p - 0.5)


def make_pipeline(alpha, l1_ratio):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("probe", ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=50000,
            tol=1e-3,
            random_state=SEED,
        )),
    ])


def get_xy(df):
    X = np.stack(df["vector"].apply(np.array).values)
    y = build_pairwise_target(df)
    return X, y


def fit_predict_probe(train_df, test_df, alpha, l1_ratio):
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)

    pipeline = make_pipeline(alpha, l1_ratio)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rho = safe_spearman(y_test, y_pred)
    coef = pipeline.named_steps["probe"].coef_
    nonzero = int(np.sum(np.abs(coef) > 1e-12))

    return rho, nonzero


def summarize_fold_results(rhos, nonzeros, n_trains, n_tests):
    return {
        "rho_mean": round(float(np.mean(rhos)), 4),
        "rho_std": round(float(np.std(rhos)), 4),
        "n_nonzero_mean": round(float(np.mean(nonzeros)), 2),
        "n_nonzero_std": round(float(np.std(nonzeros)), 2),
        "n_train": round(float(np.mean(n_trains)), 2),
        "n_test": round(float(np.mean(n_tests)), 2),
        "n_train_total": int(np.sum(n_trains)),
        "n_test_total": int(np.sum(n_tests)),
        "n_folds": int(len(rhos)),
    }


def run_cv_probe_for_subset(
    df,
    n_splits,
    alpha,
    l1_ratio,
    rep_type,
    layer,
    save_coef=False,
):
    X, y = get_xy(df)
    groups = build_groups(df)

    n_groups = len(pd.unique(groups))
    if n_groups < n_splits:
        return None, []

    pipeline = make_pipeline(alpha, l1_ratio)
    cv = GroupKFold(n_splits=n_splits)

    rhos = []
    nonzeros = []
    coef_rows = []
    n_trains = []
    n_tests = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        rho = safe_spearman(y_test, y_pred)
        rhos.append(rho)
        n_trains.append(len(train_idx))
        n_tests.append(len(test_idx))

        coef = pipeline.named_steps["probe"].coef_
        nonzero = int(np.sum(np.abs(coef) > 1e-12))
        nonzeros.append(nonzero)

        if save_coef:
            for dim, weight in enumerate(coef):
                coef_rows.append({
                    "rep_type": rep_type,
                    "layer": int(layer),
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "fold": int(fold),
                    "dim": int(dim),
                    "coef": float(weight),
                    "abs_coef": float(abs(weight)),
                    "is_nonzero": bool(abs(weight) > 1e-12),
                    "rho": float(rho),
                })

    result = summarize_fold_results(rhos, nonzeros, n_trains, n_tests)
    result["n_groups"] = int(n_groups)

    return result, coef_rows


def run_language_split_transfer(src_df, tgt_df, args):
    rho, nonzero = fit_predict_probe(
        train_df=src_df,
        test_df=tgt_df,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
    )

    return {
        "rho_mean": round(float(rho), 4),
        "rho_std": 0.0,
        "n_nonzero_mean": float(nonzero),
        "n_nonzero_std": 0.0,
        "n_train": int(len(src_df)),
        "n_test": int(len(tgt_df)),
        "n_train_total": int(len(src_df)),
        "n_test_total": int(len(tgt_df)),
        "n_folds": 1,
        "n_groups": int(len(pd.unique(tgt_df["binom_id"]))),
    }


def run_binom_id_heldout_transfer(src_df, tgt_df, args):
    src_ids = set(src_df["binom_id"].dropna().astype(int).unique().tolist())
    tgt_ids = set(tgt_df["binom_id"].dropna().astype(int).unique().tolist())
    common_ids = np.array(sorted(src_ids & tgt_ids), dtype=int)

    if len(common_ids) < args.n_splits:
        return None

    dummy_x = np.zeros((len(common_ids), 1))
    cv = GroupKFold(n_splits=args.n_splits)

    rhos = []
    nonzeros = []
    n_trains = []
    n_tests = []

    for train_idx, test_idx in cv.split(dummy_x, common_ids, groups=common_ids):
        train_ids = set(common_ids[train_idx].tolist())
        test_ids = set(common_ids[test_idx].tolist())

        train_fold = src_df[src_df["binom_id"].astype(int).isin(train_ids)].copy()
        test_fold = tgt_df[tgt_df["binom_id"].astype(int).isin(test_ids)].copy()

        if train_fold.empty or test_fold.empty:
            continue

        rho, nonzero = fit_predict_probe(
            train_df=train_fold,
            test_df=test_fold,
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
        )

        rhos.append(rho)
        nonzeros.append(nonzero)
        n_trains.append(len(train_fold))
        n_tests.append(len(test_fold))

    if not rhos:
        return None

    result = summarize_fold_results(rhos, nonzeros, n_trains, n_tests)
    result["n_groups"] = int(len(common_ids))
    return result


def run_cv_mode(df, args, model_name):
    rows = []
    all_coef_rows = []
    save_coef = args.coef_output is not None

    grouped = df.groupby(["rep_type", "layer"], sort=True)

    for (rep_type, layer), subdf in grouped:
        result, coef_rows = run_cv_probe_for_subset(
            df=subdf,
            n_splits=args.n_splits,
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
            rep_type=rep_type,
            layer=layer,
            save_coef=save_coef,
        )

        if result is None:
            continue

        rows.append({
            "model": model_name,
            "rep_type": rep_type,
            "layer": int(layer),
            "alpha": float(args.alpha),
            "l1_ratio": float(args.l1_ratio),
            "rho_mean": result["rho_mean"],
            "rho_std": result["rho_std"],
            "n_nonzero_mean": result["n_nonzero_mean"],
            "n_nonzero_std": result["n_nonzero_std"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "n_groups": result["n_groups"],
            "n_folds": result["n_folds"],
        })

        all_coef_rows.extend(coef_rows)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("No valid probe results were produced.")

    result_df = result_df.sort_values(
        ["rep_type", "rho_mean", "layer"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    return result_df, all_coef_rows


def run_crosslingual_mode(df, args, model_name):
    rows = []
    grouped = df.groupby(["rep_type", "layer"], sort=True)

    for (rep_type, layer), subdf in grouped:
        langs = sorted(subdf["lang"].dropna().unique().tolist())

        for src_lang in langs:
            src_df = subdf[subdf["lang"] == src_lang].copy()
            if src_df.empty:
                continue

            for tgt_lang in langs:
                tgt_df = subdf[subdf["lang"] == tgt_lang].copy()
                if tgt_df.empty:
                    continue

                if args.xling_split == "binom_id":
                    result = run_binom_id_heldout_transfer(src_df, tgt_df, args)
                    split_name = "heldout_binom_id"
                else:
                    if src_lang == tgt_lang:
                        result, _ = run_cv_probe_for_subset(
                            df=src_df,
                            n_splits=args.n_splits,
                            alpha=args.alpha,
                            l1_ratio=args.l1_ratio,
                            rep_type=rep_type,
                            layer=layer,
                            save_coef=False,
                        )
                    else:
                        result = run_language_split_transfer(src_df, tgt_df, args)
                    split_name = "language"

                if result is None:
                    continue

                rows.append({
                    "model": model_name,
                    "mode": "within_language_cv" if src_lang == tgt_lang else "crosslingual",
                    "split": split_name,
                    "rep_type": rep_type,
                    "layer": int(layer),
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "alpha": float(args.alpha),
                    "l1_ratio": float(args.l1_ratio),
                    "rho_mean": result["rho_mean"],
                    "rho_std": result["rho_std"],
                    "n_nonzero_mean": result["n_nonzero_mean"],
                    "n_nonzero_std": result["n_nonzero_std"],
                    "n_train": result["n_train"],
                    "n_test": result["n_test"],
                    "n_train_total": result["n_train_total"],
                    "n_test_total": result["n_test_total"],
                    "n_groups": result["n_groups"],
                    "n_folds": result["n_folds"],
                })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("No valid cross-lingual probe results were produced.")

    result_df = result_df.sort_values(
        ["rep_type", "layer", "src_lang", "tgt_lang"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    return result_df


def main():
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(in_path)
    df = df[df["source"] == "hidden"].copy()

    if args.rep_types is not None:
        df = df[df["rep_type"].isin(args.rep_types)].copy()

    if args.langs is not None:
        df = df[df["lang"].isin(args.langs)].copy()

    if args.layers is not None:
        df = df[df["layer"].isin(args.layers)].copy()

    if args.tier is not None:
        if "evidence_tier" not in df.columns:
            raise ValueError("Column 'evidence_tier' not found, but --tier was provided.")
        df = df[df["evidence_tier"] == args.tier].copy()

    if df.empty:
        raise ValueError("No rows left after filtering.")

    model_name = df["model"].iloc[0] if "model" in df.columns else "unknown"
    probe_name = "lasso" if args.l1_ratio == 1.0 else "elasticnet"
    rep_types_used = sorted(df["rep_type"].unique().tolist())

    print("\n[PROBE CONFIG]")
    print(f"model     : {model_name}")
    print(f"mode      : {args.mode}")
    print(f"probe     : {probe_name}")
    print("target    : abs_p_corpus_minus_0.5")
    print(f"rep_types : {rep_types_used}")
    print(f"alpha     : {args.alpha}")
    print(f"l1_ratio  : {args.l1_ratio}")
    print(f"n_splits  : {args.n_splits}")
    print(f"seed      : {SEED}")

    if args.mode == "cv":
        print("grouping  : lang+binom_id")
        result_df, all_coef_rows = run_cv_mode(df, args, model_name)
    else:
        print("transfer  : source_language -> target_language")
        print(f"xling_split: {args.xling_split}")
        if args.xling_split == "binom_id":
            print("split     : held-out binomial IDs for every src->tgt pair")
        else:
            print("split     : train on all source-language rows, test on all target-language rows")
        result_df = run_crosslingual_mode(df, args, model_name)
        all_coef_rows = []
        if args.coef_output is not None:
            print("[WARN] --coef_output is ignored in --mode crosslingual.")

    print("\n[PROBE RESULTS]")
    print(result_df.to_string(index=False))

    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[DONE] Results saved to {out_path}")

    if args.mode == "cv" and args.coef_output is not None:
        coef_path = Path(args.coef_output)
        coef_path.parent.mkdir(parents=True, exist_ok=True)

        coef_df = pd.DataFrame(all_coef_rows)
        if coef_df.empty:
            raise ValueError("No coefficients were saved.")

        coef_df = coef_df.sort_values(
            ["rep_type", "layer", "fold", "abs_coef"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

        coef_df.to_csv(coef_path, index=False, encoding="utf-8")
        print(f"[DONE] Coefficients saved to {coef_path}")


if __name__ == "__main__":
    main()