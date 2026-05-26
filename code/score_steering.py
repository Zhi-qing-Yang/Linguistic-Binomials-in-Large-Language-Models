import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prefixes import PREFIX_SETS
from utils.modeling import get_decoder_layers
from utils.binomial import build_binomial_surface
from utils.scoring import continuation_logprob
from utils.metrics import compute_behavioral_metrics
from utils.meta import load_meta


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

LANGS = ["en", "de", "ru", "zh", "ja", "tr", "id", "ar"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--steering_vec", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[-5.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
    )
    p.add_argument("--langs", nargs="+", default=LANGS)
    p.add_argument(
        "--prefix_sets",
        nargs="+",
        default=["discourse", "frequency", "metalinguistic", "minimal"],
        choices=list(PREFIX_SETS.keys()),
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))




def make_steering_hook(vec_tensor, scale):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            delta = scale * vec_tensor.to(h.device, h.dtype)
            h = h + delta.view(1, 1, -1)
            return (h,) + output[1:]

        h = output
        delta = scale * vec_tensor.to(h.device, h.dtype)
        return h + delta.view(1, 1, -1)

    return hook



def score_one(model, tokenizer, device, lang, meta_row, prefix):
    a = meta_row["a"]
    b = meta_row["b"]
    conj = meta_row["conj"]

    text_ab = build_binomial_surface(a, conj, b, lang)
    text_ba = build_binomial_surface(b, conj, a, lang)

    s_ab = continuation_logprob(model, tokenizer, prefix, text_ab, device)
    s_ba = continuation_logprob(model, tokenizer, prefix, text_ba, device)

    return {
        "s_AB": round(s_ab, 6),
        "s_BA": round(s_ba, 6),
        "p_llm": round(sigmoid(s_ab - s_ba), 6),
        "text_AB": text_ab,
        "text_BA": text_ba,
    }


def compute_metrics(records):
    return compute_behavioral_metrics(
        records,
        include_mean_p_llm=True,
        include_n=True,
        nan_rho_to_zero=True,
    )


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_summary_rows(summary_rows, args, scale, all_records):
    overall = compute_metrics(all_records)
    if overall is not None:
        summary_rows.append({
            "scale": float(scale),
            "group": "ALL_PREFIXES_ALL_LANGS",
            "lang": "ALL",
            "prefix_set": "ALL",
            **overall,
        })

        print(
            f"[ALL PREFIXES / ALL LANGS] "
            f"rho={overall['rho']:.4f} "
            f"mae={overall['mae']:.4f} "
            f"jsd={overall['jsd']:.4f} "
            f"mean_p_llm={overall['mean_p_llm']:.4f} "
            f"n={overall['n']}"
        )

    for lang in args.langs:
        lang_records = [r for r in all_records if r["lang"] == lang]
        metrics = compute_metrics(lang_records)
        if metrics is None:
            continue

        summary_rows.append({
            "scale": float(scale),
            "group": "ALL_PREFIXES_BY_LANG",
            "lang": lang,
            "prefix_set": "ALL",
            **metrics,
        })

        print(
            f"[ALL PREFIXES / {lang}] "
            f"rho={metrics['rho']:.4f} "
            f"mae={metrics['mae']:.4f} "
            f"jsd={metrics['jsd']:.4f} "
            f"mean_p_llm={metrics['mean_p_llm']:.4f} "
            f"n={metrics['n']}"
        )

    for prefix_name in args.prefix_sets:
        prefix_records = [r for r in all_records if r["prefix_set"] == prefix_name]
        metrics = compute_metrics(prefix_records)
        if metrics is None:
            continue

        summary_rows.append({
            "scale": float(scale),
            "group": "BY_PREFIX_ALL_LANGS",
            "lang": "ALL",
            "prefix_set": prefix_name,
            **metrics,
        })


def main():
    args = parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cpu":
        torch_dtype = torch.float32
    else:
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]

    print(f"[INFO] ROOT        : {ROOT}")
    print(f"[INFO] DATA_DIR    : {data_dir}")
    print(f"[INFO] model       : {args.model}")
    print(f"[INFO] layer       : {args.layer}")
    print(f"[INFO] prefix_sets : {args.prefix_sets}")
    print(f"[INFO] scales      : {args.scales}")
    print(f"[INFO] device      : {device}")
    print(f"[INFO] dtype       : {torch_dtype}")

    vec_np = np.load(args.steering_vec).astype(np.float32)
    vec_tensor = torch.from_numpy(vec_np)

    print(f"[INFO] steering vec: {args.steering_vec}")
    print(f"[INFO] vec shape   : {vec_tensor.shape}")
    print(f"[INFO] vec norm    : {float(torch.norm(vec_tensor)):.6f}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if device.type == "cpu":
        model = model.to(device)

    model.eval()

    layers = get_decoder_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"Invalid layer={args.layer}; model has {len(layers)} layers.")

    hidden_size = model.config.hidden_size
    if vec_tensor.numel() != hidden_size:
        raise ValueError(
            f"Vector dim={vec_tensor.numel()}, but model hidden_size={hidden_size}"
        )

    target_layer = layers[args.layer]

    print(f"[INFO] total layers: {len(layers)}")
    print(f"[INFO] hidden size : {hidden_size}")

    meta_by_lang = {}
    for lang in args.langs:
        meta_by_lang[lang] = load_meta(lang, data_dir)
        print(f"[INFO] loaded {lang}: {len(meta_by_lang[lang])} items")

    expected_rows = len(args.prefix_sets) * sum(len(meta_by_lang[lang]) for lang in args.langs)
    print(f"[INFO] expected rows per scale: {expected_rows}")

    model_tag = Path(args.model).name
    summary_rows = []

    for scale in args.scales:
        scale_name = f"{scale:+.2f}".replace("+", "p").replace("-", "n").replace(".", "_")
        scale_dir = out_dir / f"scale_{scale_name}"
        scale_dir.mkdir(parents=True, exist_ok=True)

        merged_jsonl = scale_dir / "scored_all_prefixes.jsonl"

        if merged_jsonl.exists():
            with open(merged_jsonl, "r", encoding="utf-8") as f:
                n_existing = sum(1 for _ in f)

            if n_existing == expected_rows:
                print(f"[SKIP] scale {scale:+.3f} already completed: {n_existing} rows")
                all_records = load_jsonl(merged_jsonl)
                append_summary_rows(summary_rows, args, scale, all_records)
                continue

            print(
                f"[WARN] scale {scale:+.3f} incomplete or invalid: "
                f"{n_existing}/{expected_rows} rows. Rerunning this scale."
            )
            shutil.rmtree(scale_dir)
            scale_dir.mkdir(parents=True, exist_ok=True)
            merged_jsonl = scale_dir / "scored_all_prefixes.jsonl"

        print("\n" + "=" * 80)
        print(f"[SCALE] {scale:+.3f}")
        print("=" * 80)

        handle = None
        if scale != 0.0:
            handle = target_layer.register_forward_hook(
                make_steering_hook(vec_tensor, scale)
            )

        all_records = []

        with open(merged_jsonl, "w", encoding="utf-8") as fout:
            for prefix_name in args.prefix_sets:
                prefix_map = PREFIX_SETS[prefix_name]

                for lang in args.langs:
                    meta = meta_by_lang[lang]
                    prefix = prefix_map[lang]

                    desc = f"{prefix_name}/{lang} scale={scale:+.1f}"

                    for binom_id in tqdm(sorted(meta.keys()), desc=desc):
                        row = meta[binom_id]
                        result = score_one(model, tokenizer, device, lang, row, prefix)

                        record = {
                            "binom_id": int(binom_id),
                            "lang": lang,
                            "model": model_tag,
                            "prefix_set": prefix_name,
                            "scale": float(scale),
                            "layer": int(args.layer),
                            **result,
                            "p_corpus": row["p_corpus"],
                            "evidence_tier": row["evidence_tier"],
                            "preferred_order": row["preferred_order"],
                        }

                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        all_records.append(record)

        if handle is not None:
            handle.remove()

        append_summary_rows(summary_rows, args, scale, all_records)

        print(f"[DONE] wrote merged records: {merged_jsonl}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"\n[DONE] summary saved to {summary_path}")

    if not summary_df.empty:
        main_summary = summary_df[
            summary_df["group"] == "ALL_PREFIXES_ALL_LANGS"
        ].copy()

        print("\n[STRICT MAIN SUMMARY: all prefixes merged directly]")
        print(main_summary[[
            "scale", "rho", "acc", "mae", "jsd", "mean_p_llm", "n"
        ]].to_string(index=False))

        lang_summary = summary_df[
            summary_df["group"] == "ALL_PREFIXES_BY_LANG"
        ].copy()

        print("\n[RHO by scale and language, all prefixes merged]")
        print(
            lang_summary
            .pivot(index="scale", columns="lang", values="rho")
            .to_string()
        )

        print("\n[MAE by scale and language, all prefixes merged]")
        print(
            lang_summary
            .pivot(index="scale", columns="lang", values="mae")
            .to_string()
        )


if __name__ == "__main__":
    main()