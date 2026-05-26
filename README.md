# Behavioral and Representational Evidence of Binomial Ordering Preferences in Large Language Models

## Setup

```bash
pip install -r requirements.txt
```

## Data

Input files and corpus counts are in `data/`:

```text
data/<lang>/input_<lang>.csv
data/<lang>/output_<lang>_sk.csv
```

## Main Evaluation

Score binomials:

```bash
python code/score_binomials.py \
  --model <model_name_or_path> \
  --output outputs/<model_family>/<size>/<prefix_set>/scored.jsonl \
  --prefix_set <prefix_set>
```

Evaluate scores:

```bash
python code/evaluate_scores.py \
  --inputs outputs/<model_family>/<size>/*/scored.jsonl \
  --output outputs/<model_family>/<size>/main.csv
```

Build the main table:

```bash
python code/make_main_table.py
```

The resulting table is written to:

```text
outputs/model_tables/main_table.csv
```

## Probing

Train probes:

```bash
python code/train_probe.py \
  --input outputs/qwen3/4b/reps.jsonl \
  --output outputs/probing/cv_lasso_003.csv \
  --mode cv \
  --alpha 0.03 \
  --l1_ratio 1.0
```

The provided probing summaries are in:

```text
outputs/probing/
```

## Steering

Extract a steering vector:

```bash
python code/extract_steering_vec.py \
  --coef outputs/probing/cv_lasso_003_coef.csv \
  --layer 14 \
  --rep_type last \
  --output outputs/steering/steer_vec.npy
```

Score with steering:

```bash
python code/score_steering.py \
  --model <model_name_or_path> \
  --steering_vec outputs/steering/steer_vec.npy \
  --layer 14 \
  --output_dir outputs/steering/<run_name>
```

Provided steering outputs are in:

```text
outputs/steering/last14/
outputs/steering/mean23/
```
