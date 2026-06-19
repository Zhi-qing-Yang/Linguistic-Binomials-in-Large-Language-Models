# Behavioral and Representational Evidence of Binomial Ordering Preferences in Large Language Models

This repository contains the data, model outputs, and analysis scripts for the binomial ordering preference experiments.

## Setup

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
code/      Scripts for scoring, evaluation, probing, and steering analyses
data/      Binomial inputs and corpus-derived order counts by language
outputs/   Model scores, evaluation summaries, probing summaries, and steering outputs
```

## Data

Input files and corpus counts are organized by language:

```text
data/<lang>/input_<lang>.csv
data/<lang>/output_<lang>_sk.csv
```

The repository includes data for English, German, Russian, Indonesian, Arabic, Turkish, Japanese and Chinese.

| File | Description |
| --- | --- |
| `input_<lang>.csv` | Input binomial pairs for each language |
| `output_<lang>_sk.csv` | Binomial pairs with Sketch Engine corpus counts, smoothed corpus preference, and evidence tier |

Corpus count extraction and post-processing are implemented in:

```text
code/count_sketchengine_binomials.py
code/compute_corpus_stats.py
```

## Models

The provided outputs cover the following model checkpoints:

| Model label | Hugging Face checkpoint |
| --- | --- |
| Qwen3-4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| Qwen3-14B | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| Gemma-3-4B-IT | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) |
| Gemma-3-12B-IT | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) |

Please refer to the corresponding model releases or model cards for model-specific details and licensing terms.

## Behavioral Evaluation

The aggregated behavioral evaluation table is provided at:

```text
outputs/model_tables/main_table.csv
```

Per-model evaluation summaries are stored as:

```text
outputs/<model_dir>/<size_dir>/main.csv
```

The current output directories are `qwen3/4b`, `qwen3/14b`, `llama/3b`, `llama/8b`, `gemma/4b`, and `gemma/12b`.

To score binomials with a model:

```bash
python code/score_binomials.py \
  --model <model_name_or_path> \
  --output outputs/<model_dir>/<size_dir>/<prefix_set>/scored.jsonl \
  --prefix_set <prefix_set>
```

To evaluate scored outputs:

```bash
python code/evaluate_scores.py \
  --inputs outputs/<model_dir>/<size_dir>/*/scored.jsonl \
  --output outputs/<model_dir>/<size_dir>/main.csv
```

The helper script `code/make_main_table.py` aggregates the provided per-model summaries into this table:

```bash
python code/make_main_table.py
```

## Representational and Steering Analyses

Probing summaries are provided in:

```text
outputs/probing/
```

Example probing command:

```bash
python code/train_probe.py \
  --input outputs/qwen3/4b/reps.jsonl \
  --output outputs/probing/cv_lasso_003.csv \
  --mode cv \
  --alpha 0.03 \
  --l1_ratio 1.0
```

Steering outputs are provided in:

```text
outputs/steering/last14/
outputs/steering/mean23/
```

Example steering vector extraction:

```bash
python code/extract_steering_vec.py \
  --coef outputs/probing/cv_lasso_003_coef.csv \
  --layer 14 \
  --rep_type last \
  --output outputs/steering/steer_vec.npy
```

Example steering run:

```bash
python code/score_steering.py \
  --model <model_name_or_path> \
  --steering_vec outputs/steering/steer_vec.npy \
  --layer 14 \
  --output_dir outputs/steering/<run_name>
```
