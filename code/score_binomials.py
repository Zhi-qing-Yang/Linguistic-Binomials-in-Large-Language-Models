import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prefixes import PREFIX_SETS
from utils.binomial import build_binomial_surface
from utils.scoring import continuation_logprob
from utils.meta import load_meta

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

LANGS = ["en", "de", "ru", "zh", "ja", "tr", "id", "ar"]

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))




def binomial_score(
    model,
    tokenizer,
    device: torch.device,
    prefix: str,
    first: str,
    conj: str,
    second: str,
    lang: str,
) -> tuple[float, str]:
    full_surface = build_binomial_surface(first, conj, second, lang)
    score = continuation_logprob(model, tokenizer, prefix, full_surface, device)
    return score, full_surface



def score_canonical(
    model,
    tokenizer,
    device,
    lang: str,
    meta_row: dict,
    prefix: str,
) -> dict:
    a, b, conj = meta_row["a"], meta_row["b"], meta_row["conj"]

    s_ab, text_ab = binomial_score(model, tokenizer, device, prefix, a, conj, b, lang)
    s_ba, text_ba = binomial_score(model, tokenizer, device, prefix, b, conj, a, lang)

    return {
        "s_AB": round(s_ab, 6),
        "s_BA": round(s_ba, 6),
        "p_llm": round(sigmoid(s_ab - s_ba), 6),
        "text_AB": text_ab,
        "text_BA": text_ba,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score full-string binomial order preferences across languages."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--langs", nargs="+", default=LANGS)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prefix_set",
        default="minimal",
        choices=list(PREFIX_SETS.keys()),
        help="Which prefix template set to use.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite the output file instead of appending only missing entries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cpu":
        torch_dtype = torch.float32
        if args.dtype != "float32":
            print("[INFO] CPU detected; overriding dtype to float32.")
    else:
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]

    print(f"[INFO] Model      : {args.model}")
    print(f"[INFO] Prefix set : {args.prefix_set}")
    print(f"[INFO] Device     : {device}")
    print(f"[INFO] Torch dtype: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
    )
    if not tokenizer.is_fast:
        print("[WARN] Fast tokenizer not available.")

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

    if hasattr(model, "hf_device_map"):
        print(f"[INFO] hf_device_map: {model.hf_device_map}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    write_mode = "w" if args.overwrite else "a"

    if out_path.exists() and not args.overwrite:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done_keys.add((obj["binom_id"], obj["lang"], obj.get("prefix_set", "")))
        print(f"[INFO] Resuming; {len(done_keys)} entries already written.")

    model_tag = Path(args.model).name
    selected_prefixes = PREFIX_SETS[args.prefix_set]

    with open(out_path, write_mode, encoding="utf-8") as fout:
        for lang in args.langs:
            print(f"\n========== {lang.upper()} ==========")

            try:
                meta = load_meta(lang, DATA_DIR)
            except FileNotFoundError as e:
                print(f"[SKIP] {e}")
                continue

            prefix = selected_prefixes[lang]

            for binom_id in tqdm(sorted(meta), desc=f"{lang}/{args.prefix_set}"):
                key = (binom_id, lang, args.prefix_set)
                if key in done_keys:
                    continue

                meta_row = meta[binom_id]

                try:
                    score = score_canonical(model, tokenizer, device, lang, meta_row, prefix)
                except Exception as e:
                    print(f"[WARN] binom_id={binom_id} lang={lang}: {e}")
                    continue

                record = {
                    "binom_id": binom_id,
                    "lang": lang,
                    "model": model_tag,
                    "prefix_set": args.prefix_set,
                    **score,
                    "p_corpus": meta_row["p_corpus"],
                    "evidence_tier": meta_row["evidence_tier"],
                    "preferred_order": meta_row["preferred_order"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                done_keys.add(key)

    print(f"\n[DONE] Results written to {out_path}")


if __name__ == "__main__":
    main()