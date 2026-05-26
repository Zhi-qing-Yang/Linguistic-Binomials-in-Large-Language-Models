import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        choices=["all", "strong", "medium", "weak"],
        default="all",
        help="Which table version to load",
    )
    return parser.parse_args()


def get_filename(tier):
    if tier == "all":
        return "main.csv"
    return "main_{}.csv".format(tier)


def get_output_name(tier):
    if tier == "all":
        return "main_table.csv"
    return "main_table_{}.csv".format(tier)


def main():
    args = parse_args()
    filename = get_filename(args.tier)

    models = {
        "Qwen3-4B": OUTPUTS / "qwen3" / "4b" / filename,
        "Qwen3-14B": OUTPUTS / "qwen3" / "14b" / filename,
        "Llama-3.2-3B": OUTPUTS / "llama" / "3b" / filename,
        "Llama-3.1-8B": OUTPUTS / "llama" / "8b" / filename,
        "Gemma-3-4B": OUTPUTS / "gemma" / "4b" / filename,
        "Gemma-3-12B": OUTPUTS / "gemma" / "12b" / filename,
    }

    lang_order = ["ar", "de", "en", "id", "ja", "ru", "tr", "zh"]
    lang_labels = {
        "ar": "AR",
        "de": "DE",
        "en": "EN",
        "id": "ID",
        "ja": "JA",
        "ru": "RU",
        "tr": "TR",
        "zh": "ZH",
    }

    rows = []

    for model_name, path in models.items():
        df = pd.read_csv(path).set_index("lang")

        row = {"model": model_name}
        rho_vals = []
        acc_vals = []
        mae_vals = []
        jsd_vals = []

        for lang in lang_order:
            label = lang_labels[lang]

            rho = round(float(df.loc[lang, "rho"]), 4)
            acc = round(float(df.loc[lang, "acc"]), 4)
            mae = round(float(df.loc[lang, "mae"]), 4)
            jsd = round(float(df.loc[lang, "jsd"]), 4)

            row["{}_rho".format(label)] = rho
            row["{}_acc".format(label)] = acc
            row["{}_mae".format(label)] = mae
            row["{}_jsd".format(label)] = jsd

            rho_vals.append(rho)
            acc_vals.append(acc)
            mae_vals.append(mae)
            jsd_vals.append(jsd)

        row["avg_rho"] = round(sum(rho_vals) / len(rho_vals), 4)
        row["avg_acc"] = round(sum(acc_vals) / len(acc_vals), 4)
        row["avg_mae"] = round(sum(mae_vals) / len(mae_vals), 4)
        row["avg_jsd"] = round(sum(jsd_vals) / len(jsd_vals), 4)

        rows.append(row)

    main_table = pd.DataFrame(rows)

    print("\n[MAIN TABLE - {}]".format(args.tier.upper()))
    print("Shape: {} rows x {} cols".format(main_table.shape[0], main_table.shape[1]))

    out_dir = OUTPUTS / "model_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / get_output_name(args.tier)
    main_table.to_csv(out_path, index=False, encoding="utf-8")

    print("\n[DONE] Saved to {}".format(out_path))


if __name__ == "__main__":
    main()
