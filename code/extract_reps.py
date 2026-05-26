import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.io import load_many_jsonl
from utils.prefixes import PREFIX_SETS
from utils.modeling import get_decoder_layers

ROOT = Path(__file__).resolve().parents[1]



def get_continuation_token_positions(tokenizer, full_text, prefix):
    prefix_char_len = len(prefix)

    enc = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets = enc["offset_mapping"][0].tolist()

    cont_token_positions = []
    for tok_idx, (start, end) in enumerate(offsets):
        if end <= prefix_char_len:
            continue
        if start == end:
            continue
        cont_token_positions.append(tok_idx)

    return enc, cont_token_positions


def pool_last(hidden, token_positions):
    pos = token_positions[-1]
    return hidden[0, pos, :].detach().float().cpu()


def pool_mean(hidden, token_positions):
    pos_tensor = torch.tensor(token_positions, device=hidden.device)
    span = hidden[0, pos_tensor, :]
    return span.mean(dim=0).detach().float().cpu()


def build_examples_from_scored(df):
    rows = []

    for _, row in df.iterrows():
        lang = row["lang"]
        prefix_set = row["prefix_set"]
        prefix = PREFIX_SETS[prefix_set][lang]

        rows.append({
            "binom_id": int(row["binom_id"]),
            "lang": lang,
            "model": row["model"],
            "prefix_set": prefix_set,
            "preferred_order": row["preferred_order"],
            "p_corpus": float(row["p_corpus"]),
            "evidence_tier": row["evidence_tier"] if "evidence_tier" in row else None,
            "order": "AB",
            "prefix": prefix,
            "continuation": row["text_AB"],
        })

        rows.append({
            "binom_id": int(row["binom_id"]),
            "lang": lang,
            "model": row["model"],
            "prefix_set": prefix_set,
            "preferred_order": row["preferred_order"],
            "p_corpus": float(row["p_corpus"]),
            "evidence_tier": row["evidence_tier"] if "evidence_tier" in row else None,
            "order": "BA",
            "prefix": prefix,
            "continuation": row["text_BA"],
        })

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug_print", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    input_paths = [Path(x) for x in args.inputs]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    print("[INFO] model      :", args.model)
    print("[INFO] device     :", device)
    print("[INFO] dtype      :", torch_dtype)
    print("[INFO] inputs     :", len(input_paths))
    print("[INFO] output     :", out_path)

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
    print("[INFO] num_layers :", len(layers))

    scored_df = load_many_jsonl(input_paths)
    if "condition" in scored_df.columns:
        scored_df = scored_df[scored_df["condition"] == "canonical"].copy()

    if args.langs is not None:
        scored_df = scored_df[scored_df["lang"].isin(args.langs)].copy()

    if scored_df.empty:
        raise ValueError("No rows left after filtering.")

    print("[INFO] scored rows :", len(scored_df))
    if args.langs is not None:
        print("[INFO] langs      :", args.langs)

    examples = build_examples_from_scored(scored_df)

    done_keys = set()
    write_mode = "w" if args.overwrite else "a"

    if out_path.exists() and not args.overwrite:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done_keys.add((
                    obj["binom_id"],
                    obj["lang"],
                    obj["prefix_set"],
                    obj["order"],
                    obj["layer"],
                    obj["rep_type"],
                ))
        print("[INFO] resuming; {} records already written.".format(len(done_keys)))

    with open(out_path, write_mode, encoding="utf-8") as fout:
        for ex_idx, ex in enumerate(examples):
            full_text = ex["prefix"] + ex["continuation"]
            enc, cont_positions = get_continuation_token_positions(
                tokenizer=tokenizer,
                full_text=full_text,
                prefix=ex["prefix"],
            )

            if not cont_positions:
                print("[WARN] no continuation tokens for", ex["binom_id"], ex["lang"], ex["order"])
                continue

            input_ids = enc["input_ids"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states

            if args.debug_print:
                token_ids = input_ids[0].tolist()
                pieces = tokenizer.convert_ids_to_tokens(token_ids)
                print("\n" + "=" * 80)
                print("EXAMPLE", ex_idx)
                print("binom_id      :", ex["binom_id"])
                print("lang          :", ex["lang"])
                print("prefix_set    :", ex["prefix_set"])
                print("order         :", ex["order"])
                print("full_text     :", full_text)
                print("cont_positions:", cont_positions)
                for pos in cont_positions:
                    print("  pos={} piece={!r} decoded={!r}".format(
                        pos,
                        pieces[pos],
                        tokenizer.decode([token_ids[pos]])
                    ))

            for layer_idx in range(len(layers)):
                layer_hidden = hidden_states[layer_idx + 1]

                hidden_last = pool_last(layer_hidden, cont_positions)
                hidden_mean = pool_mean(layer_hidden, cont_positions)

                rec_hidden_last = {
                    "binom_id": ex["binom_id"],
                    "lang": ex["lang"],
                    "model": ex["model"],
                    "prefix_set": ex["prefix_set"],
                    "preferred_order": ex["preferred_order"],
                    "p_corpus": ex["p_corpus"],
                    "evidence_tier": ex["evidence_tier"],
                    "order": ex["order"],
                    "layer": layer_idx,
                    "source": "hidden",
                    "rep_type": "last",
                    "vector": hidden_last.tolist(),
                }
                key_last = (
                    rec_hidden_last["binom_id"],
                    rec_hidden_last["lang"],
                    rec_hidden_last["prefix_set"],
                    rec_hidden_last["order"],
                    rec_hidden_last["layer"],
                    rec_hidden_last["rep_type"],
                )
                if key_last not in done_keys:
                    fout.write(json.dumps(rec_hidden_last, ensure_ascii=False) + "\n")
                    done_keys.add(key_last)

                rec_hidden_mean = {
                    "binom_id": ex["binom_id"],
                    "lang": ex["lang"],
                    "model": ex["model"],
                    "prefix_set": ex["prefix_set"],
                    "preferred_order": ex["preferred_order"],
                    "p_corpus": ex["p_corpus"],
                    "evidence_tier": ex["evidence_tier"],
                    "order": ex["order"],
                    "layer": layer_idx,
                    "source": "hidden",
                    "rep_type": "mean",
                    "vector": hidden_mean.tolist(),
                }
                key_mean = (
                    rec_hidden_mean["binom_id"],
                    rec_hidden_mean["lang"],
                    rec_hidden_mean["prefix_set"],
                    rec_hidden_mean["order"],
                    rec_hidden_mean["layer"],
                    rec_hidden_mean["rep_type"],
                )
                if key_mean not in done_keys:
                    fout.write(json.dumps(rec_hidden_mean, ensure_ascii=False) + "\n")
                    done_keys.add(key_mean)

            fout.flush()

    print("\n[DONE] wrote reps to", out_path)


if __name__ == "__main__":
    main()